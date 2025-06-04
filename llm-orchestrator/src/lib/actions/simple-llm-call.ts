import { createAction, Property } from '@activepieces/pieces-framework';
import OpenAI from 'openai';

export const simpleLLMCall = createAction({
  name: 'simple_llm_call',
  displayName: 'Simple LLM Call',
  description: 'Make a simple call to OpenAI GPT models',
  props: {
    apiKey: Property.ShortText({
      displayName: 'OpenAI API Key',
      required: true,
      description: 'Your OpenAI API key'
    }),
    model: Property.ShortText({
      displayName: 'Model',
      required: true,
      defaultValue: 'gpt-4',
      description: 'OpenAI model to use (e.g., gpt-4, gpt-3.5-turbo)'
    }),
    systemPrompt: Property.LongText({
      displayName: 'System Prompt',
      required: false,
      description: 'System prompt to set the behavior of the assistant'
    }),
    userPrompt: Property.LongText({
      displayName: 'User Prompt',
      required: true,
      description: 'The main prompt/question for the LLM'
    }),
    temperature: Property.Number({
      displayName: 'Temperature',
      required: false,
      defaultValue: 0.7,
      description: 'Controls randomness (0.0 to 2.0). Lower values make output more focused and deterministic.'
    }),
    maxTokens: Property.Number({
      displayName: 'Max Tokens',
      required: false,
      defaultValue: 1000,
      description: 'Maximum number of tokens to generate in the response'
    }),
    includeUsage: Property.Checkbox({
      displayName: 'Include Usage Stats',
      required: false,
      defaultValue: true,
      description: 'Include token usage statistics in the response'
    })
  },

  async run(context) {
    const { apiKey, model, systemPrompt, userPrompt, temperature, maxTokens, includeUsage } = context.propsValue;
    
    try {
      const openai = new OpenAI({
        apiKey: apiKey
      });

      const messages: any[] = [];
      
      if (systemPrompt) {
        messages.push({
          role: 'system',
          content: systemPrompt
        });
      }
      
      messages.push({
        role: 'user',
        content: userPrompt
      });

      const response = await openai.chat.completions.create({
        model: model,
        messages: messages,
        temperature: temperature || 0.7,
        max_tokens: maxTokens || 1000
      });

      const result = {
        content: response.choices[0]?.message?.content || '',
        model: response.model,
        timestamp: new Date().toISOString(),
        usage: includeUsage ? response.usage : undefined
      };

      return {
        success: true,
        result,
        message: 'LLM call completed successfully'
      };
      
    } catch (error) {
      throw new Error(`LLM call failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }
});