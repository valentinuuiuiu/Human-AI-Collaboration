import { createAction, Property } from '@activepieces/pieces-framework';
import { LLMProviderManager } from '../common/providers';
import { PromptTemplateEngine } from '../common/template-engine';
import { LLMProviderType, LLMMessage } from '../common/types';

export const multiProviderQuery = createAction({
  name: 'multi_provider_query',
  displayName: 'Multi-Provider Query',
  description: 'Send the same prompt to multiple LLM providers and compare results',
  props: {
    providers: Property.Array({
      displayName: 'LLM Providers',
      description: 'Configure multiple LLM providers to query',
      required: true,
      properties: {
        name: Property.ShortText({
          displayName: 'Provider Name',
          required: true
        }),
        type: Property.StaticDropdown({
          displayName: 'Provider Type',
          required: true,
          options: {
            options: [
              { label: 'OpenAI', value: LLMProviderType.OPENAI },
              { label: 'Anthropic Claude', value: LLMProviderType.ANTHROPIC },
              { label: 'Azure OpenAI', value: LLMProviderType.AZURE_OPENAI },
              { label: 'Local AI', value: LLMProviderType.LOCAL_AI }
            ]
          }
        }),
        apiKey: Property.SecretText({
          displayName: 'API Key',
          required: true
        }),
        baseUrl: Property.ShortText({
          displayName: 'Base URL',
          required: false
        }),
        model: Property.ShortText({
          displayName: 'Model',
          required: true
        }),
        weight: Property.Number({
          displayName: 'Weight',
          required: false,
          defaultValue: 1,
          description: 'Weight for scoring this provider\'s response'
        })
      }
    }),

    systemPrompt: Property.LongText({
      displayName: 'System Prompt',
      required: false,
      description: 'System message to set context and behavior'
    }),

    userPrompt: Property.LongText({
      displayName: 'User Prompt Template',
      required: true,
      description: 'User message template with variables like {{variable_name}}'
    }),

    variables: Property.Object({
      displayName: 'Template Variables',
      required: false,
      description: 'Variables to substitute in the prompt template'
    }),

    options: Property.Object({
      displayName: 'Model Options',
      required: false,
      description: 'Additional options like temperature, max_tokens, etc.'
    }),

    executionMode: Property.StaticDropdown({
      displayName: 'Execution Mode',
      required: false,
      defaultValue: 'parallel',
      options: {
        options: [
          { label: 'Parallel', value: 'parallel' },
          { label: 'Sequential', value: 'sequential' }
        ]
      },
      description: 'Execute queries in parallel or sequentially'
    }),

    compareResponses: Property.Checkbox({
      displayName: 'Compare Responses',
      required: false,
      defaultValue: true,
      description: 'Generate comparison analysis of responses'
    }),

    selectBest: Property.Checkbox({
      displayName: 'Select Best Response',
      required: false,
      defaultValue: false,
      description: 'Automatically select the best response based on criteria'
    }),

    selectionCriteria: Property.Object({
      displayName: 'Selection Criteria',
      required: false,
      description: 'Criteria for selecting best response (length, relevance, etc.)'
    }),

    trackCosts: Property.Checkbox({
      displayName: 'Track Costs',
      required: false,
      defaultValue: true
    }),

    timeout: Property.Number({
      displayName: 'Timeout (seconds)',
      required: false,
      defaultValue: 30,
      description: 'Timeout for each provider call'
    })
  },

  async run(context) {
    const { 
      providers, 
      systemPrompt, 
      userPrompt, 
      variables, 
      options, 
      executionMode,
      compareResponses,
      selectBest,
      selectionCriteria,
      trackCosts,
      timeout 
    } = context.propsValue;

    const providerManager = new LLMProviderManager();
    
    // Setup providers
    for (const provider of providers) {
      providerManager.addProvider({
        name: provider.name,
        type: provider.type as LLMProviderType,
        apiKey: provider.apiKey,
        baseUrl: provider.baseUrl,
        model: provider.model
      });
    }

    // Render prompts
    const renderedUserPrompt = PromptTemplateEngine.render({
      template: userPrompt,
      variables: variables || {}
    });

    const renderedSystemPrompt = systemPrompt ? PromptTemplateEngine.render({
      template: systemPrompt,
      variables: variables || {}
    }) : undefined;

    // Prepare messages
    const messages: LLMMessage[] = [];
    if (renderedSystemPrompt) {
      messages.push({ role: 'system', content: renderedSystemPrompt });
    }
    messages.push({ role: 'user', content: renderedUserPrompt });

    const results: any[] = [];
    let totalCost = 0;

    // Execute queries
    if (executionMode === 'parallel') {
      // Parallel execution
      const promises = providers.map(async (provider) => {
        try {
          const timeoutPromise = new Promise((_, reject) => {
            setTimeout(() => reject(new Error('Timeout')), timeout * 1000);
          });

          const queryPromise = providerManager.callProvider(
            provider.name,
            provider.type as LLMProviderType,
            messages,
            provider.model,
            options || {}
          );

          const response = await Promise.race([queryPromise, timeoutPromise]);
          
          return {
            provider: provider.name,
            model: provider.model,
            weight: provider.weight || 1,
            success: true,
            ...response
          };
        } catch (error) {
          return {
            provider: provider.name,
            model: provider.model,
            weight: provider.weight || 1,
            success: false,
            error: error instanceof Error ? error.message : 'Unknown error'
          };
        }
      });

      const responses = await Promise.all(promises);
      results.push(...responses);
    } else {
      // Sequential execution
      for (const provider of providers) {
        try {
          const response = await providerManager.callProvider(
            provider.name,
            provider.type as LLMProviderType,
            messages,
            provider.model,
            options || {}
          );

          results.push({
            provider: provider.name,
            model: provider.model,
            weight: provider.weight || 1,
            success: true,
            ...response
          });
        } catch (error) {
          results.push({
            provider: provider.name,
            model: provider.model,
            weight: provider.weight || 1,
            success: false,
            error: error instanceof Error ? error.message : 'Unknown error'
          });
        }
      }
    }

    // Calculate costs
    if (trackCosts) {
      for (const result of results) {
        if (result.success && result.usage) {
          // Simplified cost calculation
          const inputCostPer1k = 0.01;
          const outputCostPer1k = 0.03;
          
          const cost = (result.usage.prompt_tokens / 1000) * inputCostPer1k +
                      (result.usage.completion_tokens / 1000) * outputCostPer1k;
          
          result.cost = cost;
          totalCost += cost;
        }
      }
    }

    // Generate comparison analysis
    let comparison = null;
    if (compareResponses) {
      const successfulResults = results.filter(r => r.success);
      
      comparison = {
        totalProviders: providers.length,
        successfulProviders: successfulResults.length,
        failedProviders: results.filter(r => !r.success).length,
        averageLatency: successfulResults.reduce((sum, r) => sum + (r.latency || 0), 0) / successfulResults.length,
        responseStats: {
          averageLength: successfulResults.reduce((sum, r) => sum + r.content.length, 0) / successfulResults.length,
          minLength: Math.min(...successfulResults.map(r => r.content.length)),
          maxLength: Math.max(...successfulResults.map(r => r.content.length))
        },
        uniqueResponses: new Set(successfulResults.map(r => r.content)).size,
        consensus: this.analyzeConsensus(successfulResults)
      };
    }

    // Select best response
    let bestResponse = null;
    if (selectBest && results.some(r => r.success)) {
      bestResponse = this.selectBestResponse(
        results.filter(r => r.success),
        selectionCriteria || {}
      );
    }

    return {
      success: true,
      prompt: {
        system: renderedSystemPrompt,
        user: renderedUserPrompt
      },
      results,
      comparison,
      bestResponse,
      summary: {
        totalProviders: providers.length,
        successfulCalls: results.filter(r => r.success).length,
        failedCalls: results.filter(r => !r.success).length,
        totalCost: trackCosts ? totalCost : undefined,
        executionMode,
        averageLatency: results
          .filter(r => r.success && r.latency)
          .reduce((sum, r) => sum + r.latency, 0) / results.filter(r => r.success).length
      }
    };
  },

  analyzeConsensus(results: any[]): any {
    if (results.length < 2) return null;

    // Simple consensus analysis
    const responses = results.map(r => r.content);
    const similarities: number[] = [];

    for (let i = 0; i < responses.length; i++) {
      for (let j = i + 1; j < responses.length; j++) {
        // Simple similarity based on common words
        const words1 = responses[i].toLowerCase().split(/\s+/);
        const words2 = responses[j].toLowerCase().split(/\s+/);
        const commonWords = words1.filter(word => words2.includes(word));
        const similarity = commonWords.length / Math.max(words1.length, words2.length);
        similarities.push(similarity);
      }
    }

    const averageSimilarity = similarities.reduce((sum, sim) => sum + sim, 0) / similarities.length;

    return {
      averageSimilarity,
      consensusLevel: averageSimilarity > 0.7 ? 'high' : averageSimilarity > 0.4 ? 'medium' : 'low',
      mostCommonWords: this.findMostCommonWords(responses)
    };
  },

  findMostCommonWords(responses: string[]): string[] {
    const wordCounts: Record<string, number> = {};
    
    responses.forEach(response => {
      const words = response.toLowerCase().split(/\s+/);
      words.forEach(word => {
        if (word.length > 3) { // Ignore short words
          wordCounts[word] = (wordCounts[word] || 0) + 1;
        }
      });
    });

    return Object.entries(wordCounts)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 10)
      .map(([word]) => word);
  },

  selectBestResponse(results: any[], criteria: any): any {
    let bestResult = results[0];
    let bestScore = 0;

    for (const result of results) {
      let score = 0;

      // Weight by provider weight
      score += result.weight || 1;

      // Consider latency (lower is better)
      if (criteria.considerLatency && result.latency) {
        const maxLatency = Math.max(...results.map(r => r.latency || 0));
        score += (maxLatency - result.latency) / maxLatency;
      }

      // Consider response length
      if (criteria.preferredLength) {
        const lengthDiff = Math.abs(result.content.length - criteria.preferredLength);
        score += Math.max(0, 1 - lengthDiff / criteria.preferredLength);
      }

      // Consider token efficiency
      if (criteria.considerTokens && result.usage) {
        const efficiency = result.content.length / result.usage.total_tokens;
        score += efficiency;
      }

      if (score > bestScore) {
        bestScore = score;
        bestResult = result;
      }
    }

    return {
      ...bestResult,
      selectionScore: bestScore,
      selectionReason: 'Highest weighted score based on criteria'
    };
  }
});