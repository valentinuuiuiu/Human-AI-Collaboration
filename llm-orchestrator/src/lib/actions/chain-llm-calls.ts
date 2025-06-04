import { createAction, Property } from '@activepieces/pieces-framework';
import { LLMProviderManager } from '../common/providers';
import { ContextManager } from '../common/context-manager';
import { PromptTemplateEngine } from '../common/template-engine';
import { LLMProviderType, LLMMessage } from '../common/types';

export const chainLLMCalls = createAction({
  name: 'chain_llm_calls',
  displayName: 'Chain LLM Calls',
  description: 'Execute multiple LLM calls in sequence with context passing',
  props: {
    providers: Property.Array({
      displayName: 'LLM Providers',
      description: 'Configure multiple LLM providers',
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
        apiKey: Property.ShortText({
          displayName: 'API Key',
          required: true
        }),
        baseUrl: Property.ShortText({
          displayName: 'Base URL',
          required: false,
          description: 'Custom base URL for the provider'
        }),
        model: Property.ShortText({
          displayName: 'Model',
          required: true,
          description: 'Model to use (e.g., gpt-4, claude-3-sonnet-20240229)'
        })
      }
    }),
    
    calls: Property.Array({
      displayName: 'LLM Calls Chain',
      description: 'Define the sequence of LLM calls',
      required: true,
      properties: {
        name: Property.ShortText({
          displayName: 'Call Name',
          required: true,
          description: 'Unique name for this call step'
        }),
        provider: Property.ShortText({
          displayName: 'Provider Name',
          required: true,
          description: 'Name of the provider to use (must match provider configuration)'
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
        useContextFrom: Property.ShortText({
          displayName: 'Use Context From',
          required: false,
          description: 'Previous call name to use context from'
        }),
        passResultAs: Property.ShortText({
          displayName: 'Pass Result As',
          required: false,
          description: 'Variable name to store this call\'s result for next calls'
        }),
        options: Property.Object({
          displayName: 'Model Options',
          required: false,
          description: 'Additional options like temperature, max_tokens, etc.'
        })
      }
    }),

    contextId: Property.ShortText({
      displayName: 'Context ID',
      required: false,
      description: 'Existing context ID to continue conversation'
    }),

    globalVariables: Property.Object({
      displayName: 'Global Variables',
      required: false,
      description: 'Variables available to all calls in the chain'
    }),

    trackCosts: Property.Checkbox({
      displayName: 'Track Costs',
      required: false,
      defaultValue: true,
      description: 'Track and report usage costs'
    }),

    failOnError: Property.Checkbox({
      displayName: 'Fail on Error',
      required: false,
      defaultValue: true,
      description: 'Stop chain execution if any call fails'
    })
  },

  async run(context) {
    const { providers, calls, contextId, globalVariables, trackCosts, failOnError } = context.propsValue;
    
    // Initialize managers
    const providerManager = new LLMProviderManager();
    const contextManager = new ContextManager();
    
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

    // Get or create context
    const workflowContextId = contextId || contextManager.createContext(globalVariables || {});
    const workflowContext = contextManager.getContext(workflowContextId);
    
    if (!workflowContext) {
      throw new Error('Failed to create or retrieve workflow context');
    }

    const results: any[] = [];
    let totalCost = 0;
    let totalTokens = 0;

    try {
      for (let i = 0; i < calls.length; i++) {
        const call = calls[i];
        const provider = providers.find(p => p.name === call.provider);
        
        if (!provider) {
          const error = `Provider ${call.provider} not found`;
          if (failOnError) {
            throw new Error(error);
          } else {
            results.push({ name: call.name, error, skipped: true });
            continue;
          }
        }

        // Prepare variables for this call
        const callVariables = {
          ...globalVariables,
          ...call.variables,
          ...workflowContext.variables
        };

        // Add results from previous calls
        results.forEach(result => {
          if (result.passResultAs) {
            callVariables[result.passResultAs] = result.content;
          }
        });

        // Render prompt template
        const renderedPrompt = PromptTemplateEngine.render({
          template: call.userPrompt,
          variables: callVariables
        });

        // Prepare messages
        const messages: LLMMessage[] = [];
        
        // Add system message if provided
        if (call.systemPrompt) {
          messages.push({
            role: 'system',
            content: PromptTemplateEngine.render({
              template: call.systemPrompt,
              variables: callVariables
            })
          });
        }

        // Add context from previous call if specified
        if (call.useContextFrom) {
          const previousResult = results.find(r => r.name === call.useContextFrom);
          if (previousResult) {
            messages.push({
              role: 'assistant',
              content: previousResult.content
            });
          }
        }

        // Add current user message
        messages.push({
          role: 'user',
          content: renderedPrompt
        });

        // Make the LLM call
        const response = await providerManager.callProvider(
          call.provider,
          provider.type as LLMProviderType,
          messages,
          provider.model,
          call.options || {}
        );

        // Calculate cost (simplified estimation)
        let callCost = 0;
        if (trackCosts && response.usage) {
          // Basic cost calculation - would need real pricing data
          const inputCostPer1k = 0.01; // $0.01 per 1k input tokens
          const outputCostPer1k = 0.03; // $0.03 per 1k output tokens
          
          callCost = (response.usage.prompt_tokens / 1000) * inputCostPer1k +
                    (response.usage.completion_tokens / 1000) * outputCostPer1k;
          
          totalCost += callCost;
          totalTokens += response.usage.total_tokens;
        }

        // Store result
        const callResult = {
          name: call.name,
          content: response.content,
          provider: response.provider,
          model: response.model,
          usage: response.usage,
          cost: callCost,
          latency: response.latency,
          passResultAs: call.passResultAs,
          metadata: response.metadata
        };

        results.push(callResult);

        // Update context
        contextManager.addMessage(workflowContextId, {
          role: 'user',
          content: renderedPrompt
        });
        
        contextManager.addMessage(workflowContextId, {
          role: 'assistant',
          content: response.content
        });

        contextManager.addResult(workflowContextId, response);

        // Store result as variable if specified
        if (call.passResultAs) {
          contextManager.setVariable(workflowContextId, call.passResultAs, response.content);
        }
      }

      return {
        success: true,
        contextId: workflowContextId,
        results,
        summary: {
          totalCalls: calls.length,
          successfulCalls: results.filter(r => !r.error).length,
          failedCalls: results.filter(r => r.error).length,
          totalCost: trackCosts ? totalCost : undefined,
          totalTokens: trackCosts ? totalTokens : undefined
        },
        context: contextManager.exportContext(workflowContextId)
      };

    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        contextId: workflowContextId,
        results,
        partialResults: true
      };
    }
  }
});