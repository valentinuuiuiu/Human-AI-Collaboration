import { createPiece, PieceAuth } from '@activepieces/pieces-framework';
import { PieceCategory } from '@activepieces/shared';
import { chainLLMCalls } from './lib/actions/chain-llm-calls';
import { multiProviderQuery } from './lib/actions/multi-provider-query';
import { contentGenerationPipeline } from './lib/actions/content-generation-pipeline';

export const llmOrchestratorAuth = PieceAuth.None();

export const llmOrchestrator = createPiece({
  displayName: 'LLM Orchestrator',
  description: 'Advanced LLM orchestration and workflow management for complex AI tasks',
  minimumSupportedRelease: '0.36.1',
  logoUrl: 'https://cdn.activepieces.com/pieces/llm-orchestrator.png',
  categories: [PieceCategory.ARTIFICIAL_INTELLIGENCE],
  auth: llmOrchestratorAuth,
  actions: [
    chainLLMCalls,
    multiProviderQuery,
    contentGenerationPipeline,
  ],
  authors: ['openhands'],
  triggers: [],
});