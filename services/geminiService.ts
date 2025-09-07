import { GoogleGenAI, Type, Modality } from "@google/genai";
import type { MemoryPalace, AnchorType, AnchorValue, Scene, QuickRecapItem } from '../types';

if (!process.env.API_KEY) {
  throw new Error("API_KEY environment variable is not set.");
}

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

const fileToBase64 = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve((reader.result as string).split(',')[1]);
    reader.onerror = (error) => reject(error);
  });
};

const textContentSchema = {
  type: Type.OBJECT,
  properties: {
    title: { type: Type.STRING, description: "The title of the memory palace, in the format 'Memory Palace: <ANCHOR NAME>'." },
    imagePrompt: { 
      type: Type.STRING, 
      description: "A single, detailed prompt for an image generation model to create a composite image of the entire memory palace. It must describe the anchor, the logical route, core style features, and the visual mnemonic for each item, referencing the exact bolded words. If the user provided an image, the prompt should describe how to edit that image to include the mnemonics." 
    },
    scenes: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          locus: { type: Type.STRING, description: "The specific location (locus) within the anchor." },
          description: { type: Type.STRING, description: "A 1-2 sentence explanation of the mnemonic scene, with the target words/phrases wrapped in double asterisks for bolding (e.g., **Item 1**)." }
        },
        required: ["locus", "description"]
      }
    },
    quickRecap: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          item: { type: Type.STRING, description: "The exact word/phrase to remember, wrapped in double asterisks." },
          locusHint: { type: Type.STRING, description: "A brief hint about the location of the item." }
        },
        required: ["item", "locusHint"]
      }
    }
  },
  required: ["title", "imagePrompt", "scenes", "quickRecap"]
};

const generateTextContent = async (anchorType: AnchorType, anchorValue: AnchorValue, list: string[]) => {
  const listToMemorize = list.map(item => `- ${item}`).join('\n');
  let anchorDetails: string;
  const modelParts: ({ text: string } | { inlineData: { mimeType: string; data: string } })[] = [];

  const systemInstruction = `You are MemoryPalace Builder, a creative mnemonic artist specializing in transforming any anchor (object, real-world place, or user-provided photo) into a highly memorable and vivid "memory palace."
  - Your job is to generate high-quality composite images and concise mnemonic scene explanations for each target word/phrase, always emphasizing accuracy and clarity.
  - Prioritize clear, memorable, and logical associations.
  - Use vivid, concrete visuals tied to specific loci. Use playful exaggeration or puns only to enhance memorability.
  - In explanations, bold the exact WORDS/PHRASES TO REMEMBER by wrapping them in double asterisks on every appearance (e.g., **word**).
  - Define a prominent, logical route and map each item to a unique, visually distinct locus along that path.
  - You must always output your response as a valid JSON object that adheres to the provided schema. Do not include any markdown formatting like \`\`\`json.`;

  if (anchorType === 'upload' && anchorValue instanceof File) {
    anchorDetails = "The user's provided photograph.";
    const base64Data = await fileToBase64(anchorValue);
    modelParts.push({
      inlineData: {
        mimeType: anchorValue.type,
        data: base64Data,
      },
    });
  } else {
    anchorDetails = anchorValue as string;
  }

  const userPrompt = `Create a memory palace based on the following anchor and list.
  
  Anchor Type: ${anchorType}
  Anchor Details: ${anchorDetails}
  List to Memorize:
  ${listToMemorize}
  
  Follow all instructions and generate the output in the specified JSON format.`;

  modelParts.push({ text: userPrompt });

  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: { parts: modelParts },
    config: {
      systemInstruction: systemInstruction,
      responseMimeType: "application/json",
      responseSchema: textContentSchema,
    },
  });

  const jsonString = response.text.trim();
  try {
    return JSON.parse(jsonString);
  } catch (e) {
    console.error("Failed to parse JSON response:", jsonString);
    throw new Error("The AI returned an invalid response. Please try again.");
  }
};

const generateImages = async (anchorType: AnchorType, anchorValue: AnchorValue, prompt: string): Promise<string[]> => {
  const generateOneImage = async (): Promise<string> => {
    const modelParts: ({ text: string } | { inlineData: { mimeType: string; data: string } })[] = [];

    if (anchorType === 'upload' && anchorValue instanceof File) {
      const base64Data = await fileToBase64(anchorValue);
      modelParts.push({
        inlineData: {
          data: base64Data,
          mimeType: anchorValue.type,
        },
      });
      modelParts.push({ text: prompt });
    } else {
      modelParts.push({ text: `${prompt}. Style: saturated color, crisp visuals, clear labels, cartoon-style where helpful, readable scene captions.` });
    }
  
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash-image-preview',
      contents: {
        parts: modelParts,
      },
      config: {
        responseModalities: [Modality.IMAGE, Modality.TEXT],
      },
    });
      
    const imagePart = response.candidates?.[0]?.content?.parts.find(part => part.inlineData);
    if (imagePart?.inlineData) {
      return `data:${imagePart.inlineData.mimeType};base64,${imagePart.inlineData.data}`;
    }
    
    const textResponse = response.candidates?.[0]?.content?.parts.find(part => part.text)?.text;
    console.error("Image generation failed. Model response:", textResponse);
    throw new Error("Failed to generate or edit the image. The model may have returned text instead.");
  };

  const imagePromises = Array(4).fill(0).map(() => generateOneImage());
  return Promise.all(imagePromises);
};

export const generateMemoryPalace = async (anchorType: AnchorType, anchorValue: AnchorValue, list: string[]): Promise<MemoryPalace> => {
  const textContent = await generateTextContent(anchorType, anchorValue, list);
  const imageUrls = await generateImages(anchorType, anchorValue, textContent.imagePrompt);

  return {
    ...textContent,
    imageGenerations: [imageUrls],
  };
};

const dataUrlToBase64 = (dataUrl: string): { mimeType: string, data: string } => {
  const parts = dataUrl.split(',');
  const mimeMatch = parts[0].match(/:(.*?);/);
  if (!mimeMatch || !mimeMatch[1] || !parts[1]) {
    throw new Error("Invalid data URL format");
  }
  const mimeType = mimeMatch[1];
  const data = parts[1];
  return { mimeType, data };
};

const editAndGenerateImages = async (baseImageUrl: string, prompt: string): Promise<string[]> => {
    const { mimeType, data: base64Data } = dataUrlToBase64(baseImageUrl);

    const editOneImage = async (): Promise<string> => {
        const modelParts = [
            {
                inlineData: {
                    data: base64Data,
                    mimeType: mimeType,
                },
            },
            { text: prompt },
        ];

        const response = await ai.models.generateContent({
            model: 'gemini-2.5-flash-image-preview',
            contents: { parts: modelParts },
            config: {
                responseModalities: [Modality.IMAGE, Modality.TEXT],
            },
        });

        const imagePart = response.candidates?.[0]?.content?.parts.find(part => part.inlineData);
        if (imagePart?.inlineData) {
            return `data:${imagePart.inlineData.mimeType};base64,${imagePart.inlineData.data}`;
        }
        
        const textResponse = response.candidates?.[0]?.content?.parts.find(part => part.text)?.text;
        console.error("Image editing failed. Model response:", textResponse);
        throw new Error("Failed to edit the image. The model may have returned text instead.");
    };

    const imagePromises = Array(4).fill(0).map(() => editOneImage());
    return Promise.all(imagePromises);
};

const regenerateTextContent = async (base64Image: { mimeType: string; data: string }, list: string[]): Promise<{title: string, imagePrompt: string, scenes: Scene[], quickRecap: QuickRecapItem[]}> => {
  const listToMemorize = list.map(item => `- ${item}`).join('\n');
  const modelParts: ({ text: string } | { inlineData: { mimeType: string; data: string } })[] = [];

  const systemInstruction = `You are MemoryPalace Builder, a creative mnemonic artist specializing in transforming any anchor (object, real-world place, or user-provided photo) into a highly memorable and vivid "memory palace."
  - Your job is to generate high-quality composite images and concise mnemonic scene explanations for each target word/phrase, always emphasizing accuracy and clarity.
  - Prioritize clear, memorable, and logical associations.
  - Use vivid, concrete visuals tied to specific loci. Use playful exaggeration or puns only to enhance memorability.
  - In explanations, bold the exact WORDS/PHRASES TO REMEMBER by wrapping them in double asterisks on every appearance (e.g., **word**).
  - Define a prominent, logical route and map each item to a unique, visually distinct locus along that path.
  - You must always output your response as a valid JSON object that adheres to the provided schema. Do not include any markdown formatting like \`\`\`json.`;
  
  modelParts.push({
    inlineData: {
      mimeType: base64Image.mimeType,
      data: base64Image.data,
    },
  });

  const userPrompt = `This is an updated image for a memory palace. Based *only* on the visual elements in this new image, create a new set of mnemonic scenes and a quick recap for the following list. Define a new logical route if necessary.
  
  List to Memorize:
  ${listToMemorize}
  
  Follow all instructions and generate the output in the specified JSON format.`;

  modelParts.push({ text: userPrompt });

  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: { parts: modelParts },
    config: {
      systemInstruction: systemInstruction,
      responseMimeType: "application/json",
      responseSchema: textContentSchema,
    },
  });

  const jsonString = response.text.trim();
  try {
    return JSON.parse(jsonString);
  } catch (e) {
    console.error("Failed to parse JSON response for regeneration:", jsonString);
    throw new Error("The AI returned an invalid response during text regeneration. Please try again.");
  }
};

export const regeneratePalace = async (baseImageUrl: string, editPrompt: string, list: string[]): Promise<Partial<MemoryPalace>> => {
  const newImageUrls = await editAndGenerateImages(baseImageUrl, editPrompt);
  
  if (!newImageUrls || newImageUrls.length === 0) {
      throw new Error("Failed to generate new images during regeneration step.");
  }

  const referenceImageUrl = newImageUrls[0];
  const { mimeType, data } = dataUrlToBase64(referenceImageUrl);

  const newTextContent = await regenerateTextContent({ mimeType, data }, list);

  return {
    ...newTextContent,
    imageGenerations: [newImageUrls],
  };
};