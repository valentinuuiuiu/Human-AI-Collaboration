import { LLMProviderManager } from '../common/providers';
import { LLMProviderType } from '../common/types';

export interface SimpleLLMCallParams {
  provider?: 'openai' | 'ollama';
  apiKey?: string;
  model?: string;
  systemPrompt?: string;
  userPrompt: string;
  temperature?: number;
  maxTokens?: number;
  includeUsage?: boolean;
}

export interface LLMCallResult {
  success: boolean;
  result: {
    content: string;
    model: string;
    timestamp: string;
    usage?: any;
  };
  message: string;
}

export async function simpleLLMCall(params: SimpleLLMCallParams): Promise<LLMCallResult> {
  const {
    provider = 'ollama',
    apiKey,
    model = provider === 'ollama' ? 'granite4:3b' : 'gpt-4',
    systemPrompt,
    userPrompt,
    temperature = 0.7,
    maxTokens = 1000,
    includeUsage = true
  } = params;

  try {
    const providerManager = new LLMProviderManager();

    // Configure provider based on type
    if (provider === 'openai') {
      if (!apiKey) {
        throw new Error('OpenAI API key is required');
      }
      providerManager.addProvider({
        name: 'openai',
        type: LLMProviderType.OPENAI,
        apiKey,
        model
      });
    } else if (provider === 'ollama') {
      providerManager.addProvider({
        name: 'ollama',
        type: LLMProviderType.LOCAL_AI,
        baseUrl: 'http://localhost:11434',
        model
      });
    } else {
      throw new Error(`Unsupported provider: ${provider}`);
    }

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

    const response = await providerManager.callProvider(
      provider,
      provider === 'openai' ? LLMProviderType.OPENAI : LLMProviderType.LOCAL_AI,
      messages,
      model,
      {
        temperature,
        max_tokens: maxTokens
      }
    );

    const result = {
      content: response.content,
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