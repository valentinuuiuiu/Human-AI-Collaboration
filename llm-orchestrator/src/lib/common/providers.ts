import { OpenAI } from 'openai';
import Anthropic from '@anthropic-ai/sdk';
import { LLMProvider, LLMMessage, LLMResponse, LLMProviderType } from './types';

// Simple Ollama client for local LLM access
class OllamaClient {
  constructor(private baseUrl: string = 'http://localhost:11434') {}

  async chat(model: string, messages: LLMMessage[], options: any = {}): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model,
        messages: messages.map(msg => ({
          role: msg.role,
          content: msg.content
        })),
        stream: false,
        ...options
      }),
    });

    if (!response.ok) {
      throw new Error(`Ollama API error: ${response.status} ${response.statusText}`);
    }

    return await response.json();
  }
}

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
      case LLMProviderType.LOCAL_AI:
        this.providers.set(config.name, new OllamaClient(config.baseUrl));
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

        case LLMProviderType.LOCAL_AI:
          response = await provider.chat(model, messages, options);

          return {
            content: response.message?.content || '',
            provider: providerName,
            model,
            usage: response.usage || {
              prompt_tokens: response.prompt_eval_count || 0,
              completion_tokens: response.eval_count || 0,
              total_tokens: (response.prompt_eval_count || 0) + (response.eval_count || 0)
            },
            latency: Date.now() - startTime,
            metadata: {
              done: response.done,
              total_duration: response.total_duration,
              load_duration: response.load_duration,
              prompt_eval_duration: response.prompt_eval_duration,
              eval_duration: response.eval_duration
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