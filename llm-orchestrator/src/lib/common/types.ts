export interface LLMProvider {
  name: string;
  apiKey: string;
  baseUrl?: string;
  model?: string;
}

export interface LLMMessage {
  role: 'system' | 'user' | 'assistant' | 'function';
  content: string;
  name?: string;
  function_call?: any;
}

export interface LLMResponse {
  content: string;
  provider: string;
  model: string;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  cost?: number;
  latency?: number;
  metadata?: Record<string, any>;
}

export interface WorkflowContext {
  id: string;
  messages: LLMMessage[];
  variables: Record<string, any>;
  results: LLMResponse[];
  metadata: Record<string, any>;
}

export interface PromptTemplate {
  template: string;
  variables: Record<string, any>;
  conditions?: Record<string, any>;
}

export interface ValidationRule {
  type: 'length' | 'format' | 'content' | 'json' | 'custom';
  criteria: any;
  errorMessage?: string;
}

export interface RetryConfig {
  maxRetries: number;
  backoffStrategy: 'linear' | 'exponential';
  baseDelay: number;
}

export interface CostConfig {
  trackCosts: boolean;
  budgetLimit?: number;
  alertThreshold?: number;
}

export enum LLMProviderType {
  OPENAI = 'openai',
  ANTHROPIC = 'anthropic',
  AZURE_OPENAI = 'azure-openai',
  LOCAL_AI = 'local-ai',
  CUSTOM = 'custom'
}

export interface MultiModalInput {
  type: 'text' | 'image' | 'audio' | 'document';
  content: string | Buffer;
  metadata?: Record<string, any>;
}