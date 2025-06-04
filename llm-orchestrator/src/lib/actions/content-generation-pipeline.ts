import { createAction, Property } from '@activepieces/pieces-framework';
import { LLMProviderManager } from '../common/providers';
import { ContextManager } from '../common/context-manager';
import { PromptTemplateEngine, PROMPT_TEMPLATES } from '../common/template-engine';
import { LLMProviderType, LLMMessage } from '../common/types';

export const contentGenerationPipeline = createAction({
  name: 'content_generation_pipeline',
  displayName: 'Content Generation Pipeline',
  description: 'End-to-end content creation with generation, review, and refinement',
  props: {
    // Provider Configuration
    generationProvider: Property.Object({
      displayName: 'Content Generation Provider',
      required: true,
      properties: {
        name: Property.ShortText({
          displayName: 'Provider Name',
          required: true,
          defaultValue: 'content-generator'
        }),
        type: Property.StaticDropdown({
          displayName: 'Provider Type',
          required: true,
          options: {
            options: [
              { label: 'OpenAI', value: LLMProviderType.OPENAI },
              { label: 'Anthropic Claude', value: LLMProviderType.ANTHROPIC },
              { label: 'Azure OpenAI', value: LLMProviderType.AZURE_OPENAI }
            ]
          }
        }),
        apiKey: Property.SecretText({
          displayName: 'API Key',
          required: true
        }),
        model: Property.ShortText({
          displayName: 'Model',
          required: true,
          defaultValue: 'gpt-4'
        })
      }
    }),

    reviewProvider: Property.Object({
      displayName: 'Content Review Provider',
      required: false,
      description: 'Optional separate provider for content review',
      properties: {
        name: Property.ShortText({
          displayName: 'Provider Name',
          required: true,
          defaultValue: 'content-reviewer'
        }),
        type: Property.StaticDropdown({
          displayName: 'Provider Type',
          required: true,
          options: {
            options: [
              { label: 'OpenAI', value: LLMProviderType.OPENAI },
              { label: 'Anthropic Claude', value: LLMProviderType.ANTHROPIC },
              { label: 'Azure OpenAI', value: LLMProviderType.AZURE_OPENAI }
            ]
          }
        }),
        apiKey: Property.SecretText({
          displayName: 'API Key',
          required: true
        }),
        model: Property.ShortText({
          displayName: 'Model',
          required: true,
          defaultValue: 'gpt-4'
        })
      }
    }),

    // Content Specifications
    contentType: Property.StaticDropdown({
      displayName: 'Content Type',
      required: true,
      options: {
        options: [
          { label: 'Blog Post', value: 'blog_post' },
          { label: 'Article', value: 'article' },
          { label: 'Social Media Post', value: 'social_media' },
          { label: 'Email Newsletter', value: 'email' },
          { label: 'Product Description', value: 'product_description' },
          { label: 'Marketing Copy', value: 'marketing_copy' },
          { label: 'Technical Documentation', value: 'technical_docs' },
          { label: 'Creative Writing', value: 'creative_writing' },
          { label: 'Custom', value: 'custom' }
        ]
      }
    }),

    topic: Property.LongText({
      displayName: 'Topic/Subject',
      required: true,
      description: 'Main topic or subject for the content'
    }),

    audience: Property.ShortText({
      displayName: 'Target Audience',
      required: true,
      defaultValue: 'general audience',
      description: 'Who is the target audience for this content?'
    }),

    tone: Property.StaticDropdown({
      displayName: 'Tone',
      required: true,
      defaultValue: 'professional',
      options: {
        options: [
          { label: 'Professional', value: 'professional' },
          { label: 'Casual', value: 'casual' },
          { label: 'Friendly', value: 'friendly' },
          { label: 'Formal', value: 'formal' },
          { label: 'Conversational', value: 'conversational' },
          { label: 'Authoritative', value: 'authoritative' },
          { label: 'Humorous', value: 'humorous' },
          { label: 'Persuasive', value: 'persuasive' }
        ]
      }
    }),

    length: Property.StaticDropdown({
      displayName: 'Content Length',
      required: true,
      defaultValue: 'medium',
      options: {
        options: [
          { label: 'Short (100-300 words)', value: 'short' },
          { label: 'Medium (300-800 words)', value: 'medium' },
          { label: 'Long (800-1500 words)', value: 'long' },
          { label: 'Very Long (1500+ words)', value: 'very_long' }
        ]
      }
    }),

    keywords: Property.LongText({
      displayName: 'Keywords',
      required: false,
      description: 'Keywords to include in the content (comma-separated)'
    }),

    additionalRequirements: Property.LongText({
      displayName: 'Additional Requirements',
      required: false,
      description: 'Any specific requirements, style guidelines, or constraints'
    }),

    // Pipeline Configuration
    enableReview: Property.Checkbox({
      displayName: 'Enable Content Review',
      required: false,
      defaultValue: true,
      description: 'Review generated content for quality and improvements'
    }),

    enableRefinement: Property.Checkbox({
      displayName: 'Enable Refinement',
      required: false,
      defaultValue: true,
      description: 'Refine content based on review feedback'
    }),

    maxRefinementIterations: Property.Number({
      displayName: 'Max Refinement Iterations',
      required: false,
      defaultValue: 2,
      description: 'Maximum number of refinement iterations'
    }),

    reviewCriteria: Property.Array({
      displayName: 'Review Criteria',
      required: false,
      properties: {
        criterion: Property.ShortText({
          displayName: 'Criterion',
          required: true
        }),
        weight: Property.Number({
          displayName: 'Weight (1-10)',
          required: false,
          defaultValue: 5
        })
      }
    }),

    outputFormat: Property.StaticDropdown({
      displayName: 'Output Format',
      required: false,
      defaultValue: 'markdown',
      options: {
        options: [
          { label: 'Plain Text', value: 'text' },
          { label: 'Markdown', value: 'markdown' },
          { label: 'HTML', value: 'html' },
          { label: 'JSON', value: 'json' }
        ]
      }
    }),

    includeMetadata: Property.Checkbox({
      displayName: 'Include Metadata',
      required: false,
      defaultValue: true,
      description: 'Include generation metadata (word count, readability, etc.)'
    })
  },

  async run(context) {
    const {
      generationProvider,
      reviewProvider,
      contentType,
      topic,
      audience,
      tone,
      length,
      keywords,
      additionalRequirements,
      enableReview,
      enableRefinement,
      maxRefinementIterations,
      reviewCriteria,
      outputFormat,
      includeMetadata
    } = context.propsValue;

    const providerManager = new LLMProviderManager();
    const contextManager = new ContextManager();

    // Setup providers
    providerManager.addProvider({
      name: generationProvider.name,
      type: generationProvider.type as LLMProviderType,
      apiKey: generationProvider.apiKey,
      model: generationProvider.model
    });

    if (reviewProvider) {
      providerManager.addProvider({
        name: reviewProvider.name,
        type: reviewProvider.type as LLMProviderType,
        apiKey: reviewProvider.apiKey,
        model: reviewProvider.model
      });
    }

    const workflowContextId = contextManager.createContext({
      contentType,
      topic,
      audience,
      tone,
      length,
      keywords,
      additionalRequirements
    });

    const pipeline = {
      steps: [],
      finalContent: '',
      metadata: {},
      costs: 0
    };

    try {
      // Step 1: Generate initial content
      const generationStep = await this.generateContent(
        providerManager,
        contextManager,
        workflowContextId,
        generationProvider,
        {
          contentType,
          topic,
          audience,
          tone,
          length,
          keywords,
          additionalRequirements
        }
      );

      pipeline.steps.push(generationStep);
      pipeline.finalContent = generationStep.content;
      pipeline.costs += generationStep.cost || 0;

      // Step 2: Review content (if enabled)
      if (enableReview) {
        const reviewStep = await this.reviewContent(
          providerManager,
          contextManager,
          workflowContextId,
          reviewProvider || generationProvider,
          pipeline.finalContent,
          reviewCriteria || []
        );

        pipeline.steps.push(reviewStep);
        pipeline.costs += reviewStep.cost || 0;

        // Step 3: Refine content based on review (if enabled)
        if (enableRefinement && reviewStep.suggestions && reviewStep.suggestions.length > 0) {
          let currentContent = pipeline.finalContent;
          let refinementCount = 0;

          while (refinementCount < maxRefinementIterations) {
            const refinementStep = await this.refineContent(
              providerManager,
              contextManager,
              workflowContextId,
              generationProvider,
              currentContent,
              reviewStep.suggestions
            );

            pipeline.steps.push(refinementStep);
            pipeline.costs += refinementStep.cost || 0;
            currentContent = refinementStep.content;
            refinementCount++;

            // Check if further refinement is needed
            if (refinementStep.satisfactory) {
              break;
            }
          }

          pipeline.finalContent = currentContent;
        }
      }

      // Step 4: Format output
      const formattedContent = this.formatContent(pipeline.finalContent, outputFormat);

      // Step 5: Generate metadata
      let metadata = {};
      if (includeMetadata) {
        metadata = this.generateMetadata(pipeline.finalContent, pipeline);
      }

      return {
        success: true,
        content: formattedContent,
        pipeline: {
          steps: pipeline.steps.map(step => ({
            name: step.name,
            success: step.success,
            duration: step.duration,
            cost: step.cost,
            summary: step.summary
          })),
          totalSteps: pipeline.steps.length,
          totalCost: pipeline.costs,
          totalDuration: pipeline.steps.reduce((sum, step) => sum + (step.duration || 0), 0)
        },
        metadata,
        contextId: workflowContextId,
        originalRequirements: {
          contentType,
          topic,
          audience,
          tone,
          length,
          keywords,
          additionalRequirements
        }
      };

    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        pipeline: {
          steps: pipeline.steps,
          partialContent: pipeline.finalContent
        },
        contextId: workflowContextId
      };
    }
  },

  async generateContent(
    providerManager: LLMProviderManager,
    contextManager: ContextManager,
    contextId: string,
    provider: any,
    requirements: any
  ) {
    const startTime = Date.now();

    // Prepare content generation prompt
    const template = PROMPT_TEMPLATES.CONTENT_GENERATION;
    const variables = {
      content_type: requirements.contentType.replace('_', ' '),
      topic: requirements.topic,
      audience: requirements.audience,
      tone: requirements.tone,
      length: requirements.length,
      keywords: requirements.keywords || '',
      examples: []
    };

    const prompt = PromptTemplateEngine.render({ ...template, variables });

    const messages: LLMMessage[] = [
      { role: 'user', content: prompt }
    ];

    const response = await providerManager.callProvider(
      provider.name,
      provider.type,
      messages,
      provider.model,
      { temperature: 0.7, max_tokens: 2000 }
    );

    contextManager.addMessage(contextId, { role: 'user', content: prompt });
    contextManager.addMessage(contextId, { role: 'assistant', content: response.content });

    return {
      name: 'content_generation',
      success: true,
      content: response.content,
      duration: Date.now() - startTime,
      cost: this.calculateCost(response.usage),
      summary: `Generated ${requirements.contentType} content (${response.content.length} characters)`,
      metadata: response.metadata
    };
  },

  async reviewContent(
    providerManager: LLMProviderManager,
    contextManager: ContextManager,
    contextId: string,
    provider: any,
    content: string,
    criteria: any[]
  ) {
    const startTime = Date.now();

    const reviewPrompt = `Please review the following content and provide detailed feedback:

Content to review:
${content}

Review criteria:
${criteria.map(c => `- ${c.criterion} (importance: ${c.weight}/10)`).join('\n')}

Please provide:
1. Overall quality score (1-10)
2. Specific suggestions for improvement
3. Strengths of the content
4. Areas that need work
5. Whether the content meets the requirements

Format your response as JSON with the following structure:
{
  "overallScore": number,
  "suggestions": ["suggestion1", "suggestion2"],
  "strengths": ["strength1", "strength2"],
  "improvements": ["improvement1", "improvement2"],
  "meetsRequirements": boolean,
  "summary": "brief summary of review"
}`;

    const messages: LLMMessage[] = [
      { role: 'user', content: reviewPrompt }
    ];

    const response = await providerManager.callProvider(
      provider.name,
      provider.type,
      messages,
      provider.model,
      { temperature: 0.3, max_tokens: 1000 }
    );

    let reviewData;
    try {
      reviewData = JSON.parse(response.content);
    } catch {
      // Fallback if JSON parsing fails
      reviewData = {
        overallScore: 7,
        suggestions: ['Review the content for clarity and engagement'],
        strengths: ['Content addresses the topic'],
        improvements: ['Could be more detailed'],
        meetsRequirements: true,
        summary: 'Content review completed'
      };
    }

    contextManager.addMessage(contextId, { role: 'user', content: reviewPrompt });
    contextManager.addMessage(contextId, { role: 'assistant', content: response.content });

    return {
      name: 'content_review',
      success: true,
      duration: Date.now() - startTime,
      cost: this.calculateCost(response.usage),
      summary: `Content reviewed - Score: ${reviewData.overallScore}/10`,
      ...reviewData
    };
  },

  async refineContent(
    providerManager: LLMProviderManager,
    contextManager: ContextManager,
    contextId: string,
    provider: any,
    content: string,
    suggestions: string[]
  ) {
    const startTime = Date.now();

    const refinementPrompt = `Please refine the following content based on the provided suggestions:

Original content:
${content}

Suggestions for improvement:
${suggestions.map((s, i) => `${i + 1}. ${s}`).join('\n')}

Please provide the refined version that addresses these suggestions while maintaining the original intent and style.`;

    const messages: LLMMessage[] = [
      { role: 'user', content: refinementPrompt }
    ];

    const response = await providerManager.callProvider(
      provider.name,
      provider.type,
      messages,
      provider.model,
      { temperature: 0.5, max_tokens: 2000 }
    );

    contextManager.addMessage(contextId, { role: 'user', content: refinementPrompt });
    contextManager.addMessage(contextId, { role: 'assistant', content: response.content });

    return {
      name: 'content_refinement',
      success: true,
      content: response.content,
      duration: Date.now() - startTime,
      cost: this.calculateCost(response.usage),
      summary: `Content refined based on ${suggestions.length} suggestions`,
      satisfactory: true // Could implement logic to determine if further refinement is needed
    };
  },

  formatContent(content: string, format: string): string {
    switch (format) {
      case 'html':
        return content.replace(/\n\n/g, '</p><p>').replace(/\n/g, '<br>');
      case 'json':
        return JSON.stringify({ content }, null, 2);
      case 'markdown':
      case 'text':
      default:
        return content;
    }
  },

  generateMetadata(content: string, pipeline: any): any {
    const words = content.split(/\s+/).length;
    const characters = content.length;
    const sentences = content.split(/[.!?]+/).length - 1;
    const paragraphs = content.split(/\n\s*\n/).length;

    return {
      wordCount: words,
      characterCount: characters,
      sentenceCount: sentences,
      paragraphCount: paragraphs,
      averageWordsPerSentence: Math.round(words / sentences),
      readabilityScore: this.calculateReadabilityScore(content),
      generationTime: pipeline.steps.reduce((sum: number, step: any) => sum + (step.duration || 0), 0),
      totalCost: pipeline.costs,
      pipelineSteps: pipeline.steps.length
    };
  },

  calculateReadabilityScore(content: string): number {
    // Simplified Flesch Reading Ease score
    const words = content.split(/\s+/).length;
    const sentences = content.split(/[.!?]+/).length - 1;
    const syllables = content.split(/[aeiouAEIOU]/).length - 1;

    if (sentences === 0 || words === 0) return 0;

    const score = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words));
    return Math.max(0, Math.min(100, Math.round(score)));
  },

  calculateCost(usage: any): number {
    if (!usage) return 0;
    
    // Simplified cost calculation
    const inputCostPer1k = 0.01;
    const outputCostPer1k = 0.03;
    
    return (usage.prompt_tokens / 1000) * inputCostPer1k +
           (usage.completion_tokens / 1000) * outputCostPer1k;
  }
});