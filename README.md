# AS_GeminiCaptioning Node


## User Guide
The AS_GeminiCaptioning node lets you generate a descriptive text prompt from an image using the Google Gemini API. Simply supply your image and adjust any optional text fields to tailor the output to your needs.

### Inputs
- **IMAGE** (Required):  
  The image you want to describe (e.g., JPEG, PNG). Connect your image input.  

- **PROMPT TYPE** (Required):  
  Choose between the preset styles "SD1.5 – SDXL" or "FLUX" to select the base style of the prompt.  
  *This determines which default reference text is used if you do not provide a custom reference.*

- **APY KEY PATH** (Required):  
  The file path to your API key needed for authenticating with the Google Gemini API.  

- **PROMPT LENGTH** (Optional):  
  An approximate word count for the final prompt.  
  *If left empty, no word count guidance will be added to the prompt.*

- **PROMPT REFERENCE** (Optional):  
  A sample text prompt format that serves as a reference for the generated description.  
  *If left empty, a default reference based on the selected PROMPT TYPE will be used.*

- **PROMPT STRUCTURE** (Optional):  
  A guideline for how the prompt should be organized (e.g., order of details such as building type, materials, location, etc.).  
  *If left empty, a standard structure will be applied automatically.*

- **IGNORE** (Optional):  
  Specific text or details that you want the node to exclude from the generated description.  
  *If left empty, nothing will be excluded from the prompt.*

- **EMPHASIS** (Optional):  
  Specific details or aspects you want to highlight in the generated description.  
  *If left empty, no additional emphasis instructions will be included.*

- **SAVE TO PATH** (Optional):  
  The directory where the generated text file should be saved.  
  *If left empty, the result will only be returned as output and not saved to a file.*

- **TXT NAME** (Optional):  
  The name for the saved text file. The node will automatically append a ".txt" extension if not provided.  
  *If left empty and a save path is provided, the file will be named "result.txt".*

### Outputs
- **RESULT PROMPT**:  
  The text response generated by the Google Gemini API.

- **REQUEST TEXT**:  
  The complete prompt text that was sent to the API.

- **LOG**:  
  A log detailing the node’s execution steps, useful for troubleshooting and reference.

## Required Libraries
- Pillow  
- requests  
- google-generativeai
- ```sh
pip install Pillow requests google-generativeai
```

