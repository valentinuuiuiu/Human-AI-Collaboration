import { LLMProviderManager } from '../common/providers';
import { LLMProviderType } from '../common/types';
import { PromptTemplateEngine, PROMPT_TEMPLATES } from '../common/template-engine';

export interface CollaborationParams {
  taskType: 'code_review' | 'debugging' | 'architecture' | 'testing' | 'optimization' | 'general';
  userQuery: string;
  context?: {
    language?: string;
    code?: string;
    error_messages?: string;
    requirements?: string;
    constraints?: string;
    current_architecture?: string;
    performance_issue?: string;
    component_name?: string;
    [key: string]: any;
  };
  provider?: 'openai' | 'ollama';
  apiKey?: string;
  model?: string;
  temperature?: number;
  maxTokens?: number;
}

export interface CollaborationResult {
  success: boolean;
  result: {
    content: string;
    model: string;
    timestamp: string;
    taskType: string;
    collaboration_style: string;
    usage?: any;
  };
  message: string;
}

export async function humanAICollaboration(params: CollaborationParams): Promise<CollaborationResult> {
  const {
    taskType,
    userQuery,
    context = {},
    provider = 'ollama',
    apiKey,
    model = provider === 'ollama' ? 'granite4:3b' : 'gpt-4',
    temperature = 0.7,
    maxTokens = 2000
  } = params;

  try {
    const providerManager = new LLMProviderManager();

    // Configure provider
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

    // Select appropriate template based on task type
    let template;
    let templateVariables;

    switch (taskType) {
      case 'code_review':
        template = PROMPT_TEMPLATES.COLLABORATION_CODE_REVIEW;
        templateVariables = {
          language: context.language || 'typescript',
          code: context.code || '',
          context: context.context || '',
          specific_concerns: context.specific_concerns || ''
        };
        break;

      case 'debugging':
        template = PROMPT_TEMPLATES.COLLABORATION_DEBUGGING;
        templateVariables = {
          issue_description: userQuery,
          error_messages: context.error_messages || '',
          reproduction_steps: context.reproduction_steps || '',
          language: context.language || 'typescript',
          code_context: context.code || ''
        };
        break;

      case 'architecture':
        template = PROMPT_TEMPLATES.COLLABORATION_ARCHITECTURE;
        templateVariables = {
          requirements: userQuery,
          constraints: context.constraints || '',
          current_architecture: context.current_architecture || '',
          scale_requirements: context.scale_requirements || ''
        };
        break;

      case 'testing':
        template = PROMPT_TEMPLATES.COLLABORATION_TESTING;
        templateVariables = {
          component_name: context.component_name || 'component',
          functionality: context.functionality || '',
          existing_tests: context.existing_tests || '',
          test_framework: context.test_framework || 'jest'
        };
        break;

      case 'optimization':
        template = PROMPT_TEMPLATES.COLLABORATION_OPTIMIZATION;
        templateVariables = {
          performance_issue: userQuery,
          current_metrics: context.current_metrics || '',
          language: context.language || 'typescript',
          code_to_optimize: context.code || '',
          constraints: context.constraints || ''
        };
        break;

      default: // general collaboration
        template = PROMPT_TEMPLATES.COLLABORATION_BASE;
        templateVariables = {
          task_type: context.task_type || 'general development task',
          constraints: context.constraints || '',
          user_query: userQuery
        };
    }

    // Render the prompt
    const systemPrompt = PromptTemplateEngine.render({
      template: template.template,
      variables: templateVariables
    });

    const messages: any[] = [
      {
        role: 'system',
        content: systemPrompt
      },
      {
        role: 'user',
        content: userQuery
      }
    ];

    // Make the LLM call
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
      taskType,
      collaboration_style: 'structured_guidance',
      usage: response.usage
    };

    return {
      success: true,
      result,
      message: `Human-AI collaboration completed for ${taskType} task`
    };

  } catch (error) {
    throw new Error(`Collaboration failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

// Convenience functions for specific collaboration types
export async function codeReviewCollaboration(
  code: string,
  language: string = 'typescript',
  context?: string,
  specificConcerns?: string
): Promise<CollaborationResult> {
  return humanAICollaboration({
    taskType: 'code_review',
    userQuery: `Please review this ${language} code`,
    context: {
      code,
      language,
      context,
      specific_concerns: specificConcerns
    }
  });
}

export async function debuggingCollaboration(
  issueDescription: string,
  context?: {
    error_messages?: string;
    reproduction_steps?: string;
    code?: string;
    language?: string;
  }
): Promise<CollaborationResult> {
  return humanAICollaboration({
    taskType: 'debugging',
    userQuery: issueDescription,
    context
  });
}

export async function architectureCollaboration(
  requirements: string,
  context?: {
    constraints?: string;
    current_architecture?: string;
    scale_requirements?: string;
  }
): Promise<CollaborationResult> {
  return humanAICollaboration({
    taskType: 'architecture',
    userQuery: requirements,
    context
  });
}

export async function testingCollaboration(
  componentName: string,
  functionality?: string,
  context?: {
    existing_tests?: string;
    test_framework?: string;
  }
): Promise<CollaborationResult> {
  return humanAICollaboration({
    taskType: 'testing',
    userQuery: `Create comprehensive tests for ${componentName}`,
    context: {
      component_name: componentName,
      functionality,
      ...context
    }
  });
}

export async function optimizationCollaboration(
  performanceIssue: string,
  context?: {
    current_metrics?: string;
    code?: string;
    language?: string;
    constraints?: string;
  }
): Promise<CollaborationResult> {
  return humanAICollaboration({
    taskType: 'optimization',
    userQuery: performanceIssue,
    context
  });
}