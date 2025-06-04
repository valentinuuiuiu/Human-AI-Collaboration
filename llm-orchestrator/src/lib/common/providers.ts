import { OpenAI } from 'openai';
import Anthropic from '@anthropic-ai/sdk';
import { LLMProvider, LLMMessage, LLMResponse, LLMProviderType } from './types';

export class LLMProviderManager {
  private providers: Map<string, any> = new Map();

  constructor() {}

  addProvider(config: LLMProvider & { type: LLMProviderType }) {
    switch (config.type) {
      case LLMProviderType.OPENAI:
        this.providers.set(config.name, new OpenAI({
          apiKey: config.apiKey,
          baseURL: config.baseUrl
        }));
        break;
      case LLMProviderType.ANTHROPIC:
        this.providers.set(config.name, new Anthropic({
          apiKey: config.apiKey
        }));
        break;
      case LLMProviderType.AZURE_OPENAI:
        this.providers.set(config.name, new OpenAI({
          apiKey: config.apiKey,
          baseURL: config.baseUrl,
          defaultQuery: { 'api-version': '2024-02-01' },
          defaultHeaders: {
            'api-key': config.apiKey,
          }
        }));
        break;
      default:
        throw new Error(`Unsupported provider type: ${config.type}`);
    }
  }

  async callProvider(
    providerName: string,
    providerType: LLMProviderType,
    messages: LLMMessage[],
    model: string,
    options: any = {}
  ): Promise<LLMResponse> {
    const provider = this.providers.get(providerName);
    if (!provider) {
      throw new Error(`Provider ${providerName} not found`);
    }

    const startTime = Date.now();

    try {
      let response: any;
      
      switch (providerType) {
        case LLMProviderType.OPENAI:
        case LLMProviderType.AZURE_OPENAI:
          response = await provider.chat.completions.create({
            model,
            messages: messages.map(msg => ({
              role: msg.role,
              content: msg.content,
              ...(msg.name && { name: msg.name }),
              ...(msg.function_call && { function_call: msg.function_call })
            })),
            ...options
          });
          
          return {
            content: response.choices[0]?.message?.content || '',
            provider: providerName,
            model,
            usage: response.usage,
            latency: Date.now() - startTime,
            metadata: {
              finish_reason: response.choices[0]?.finish_reason,
              response_id: response.id
            }
          };

        case LLMProviderType.ANTHROPIC:
          response = await provider.messages.create({
            model,
            messages: messages.filter(msg => msg.role !== 'system').map(msg => ({
              role: msg.role === 'assistant' ? 'assistant' : 'user',
              content: msg.content
            })),
            system: messages.find(msg => msg.role === 'system')?.content,
            max_tokens: options.max_tokens || 4000,
            ...options
          });

          return {
            content: response.content[0]?.text || '',
            provider: providerName,
            model,
            usage: response.usage,
            latency: Date.now() - startTime,
            metadata: {
              stop_reason: response.stop_reason,
              response_id: response.id
            }
          };

        default:
          throw new Error(`Unsupported provider type: ${providerType}`);
      }
    } catch (error) {
      throw new Error(`Provider ${providerName} error: ${error}`);
    }
  }

  getProvider(name: string) {
    return this.providers.get(name);
  }

  listProviders(): string[] {
    return Array.from(this.providers.keys());
  }
}