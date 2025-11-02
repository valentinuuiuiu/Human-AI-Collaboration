// Human-AI Collaboration LLM Orchestrator
// Pure TypeScript/Node.js implementation for maximum compatibility

// Core LLM functions
export { simpleLLMCall, type SimpleLLMCallParams, type LLMCallResult } from './lib/actions/simple-llm-call';

// Human-AI Collaboration functions
export {
  humanAICollaboration,
  codeReviewCollaboration,
  debuggingCollaboration,
  architectureCollaboration,
  testingCollaboration,
  optimizationCollaboration,
  type CollaborationParams,
  type CollaborationResult
} from './lib/actions/human-ai-collaboration';

// TODO: Convert remaining actions to pure functions
// export { chainLLMCalls } from './lib/actions/chain-llm-calls';
// export { multiProviderQuery } from './lib/actions/multi-provider-query';
// export { contentGenerationPipeline } from './lib/actions/content-generation-pipeline';

// Re-export common types and utilities
export * from './lib/common/types';
export * from './lib/common/providers';
export * from './lib/common/context-manager';
export * from './lib/common/template-engine';
export * from './lib/common/error-handler';