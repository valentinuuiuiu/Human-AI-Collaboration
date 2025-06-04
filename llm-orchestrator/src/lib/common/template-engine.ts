import { PromptTemplate } from './types';

export class PromptTemplateEngine {
  static render(template: PromptTemplate): string {
    let rendered = template.template;
    
    // Replace variables
    for (const [key, value] of Object.entries(template.variables)) {
      const regex = new RegExp(`{{\\s*${key}\\s*}}`, 'g');
      rendered = rendered.replace(regex, String(value));
    }
    
    // Handle conditional blocks
    if (template.conditions) {
      rendered = this.processConditionals(rendered, template.conditions);
    }
    
    // Handle loops
    rendered = this.processLoops(rendered, template.variables);
    
    return rendered.trim();
  }

  private static processConditionals(template: string, conditions: Record<string, any>): string {
    // Process {{#if condition}} blocks
    const ifRegex = /{{#if\s+(\w+)}}([\s\S]*?){{\/if}}/g;
    return template.replace(ifRegex, (match, conditionKey, content) => {
      return conditions[conditionKey] ? content : '';
    });
  }

  private static processLoops(template: string, variables: Record<string, any>): string {
    // Process {{#each array}} blocks
    const eachRegex = /{{#each\s+(\w+)}}([\s\S]*?){{\/each}}/g;
    return template.replace(eachRegex, (match, arrayKey, content) => {
      const array = variables[arrayKey];
      if (!Array.isArray(array)) return '';
      
      return array.map((item, index) => {
        let itemContent = content;
        // Replace {{this}} with current item
        itemContent = itemContent.replace(/{{this}}/g, String(item));
        // Replace {{@index}} with current index
        itemContent = itemContent.replace(/{{@index}}/g, String(index));
        // If item is object, replace {{property}} with item.property
        if (typeof item === 'object' && item !== null) {
          for (const [key, value] of Object.entries(item)) {
            const regex = new RegExp(`{{${key}}}`, 'g');
            itemContent = itemContent.replace(regex, String(value));
          }
        }
        return itemContent;
      }).join('');
    });
  }

  static validateTemplate(template: string): { valid: boolean; errors: string[] } {
    const errors: string[] = [];
    
    // Check for unclosed tags
    const openTags = template.match(/{{#\w+/g) || [];
    const closeTags = template.match(/{{\/\w+}}/g) || [];
    
    if (openTags.length !== closeTags.length) {
      errors.push('Mismatched opening and closing tags');
    }
    
    // Check for invalid syntax
    const invalidSyntax = template.match(/{{[^}]*[^}]$/g);
    if (invalidSyntax) {
      errors.push('Invalid template syntax found');
    }
    
    return {
      valid: errors.length === 0,
      errors
    };
  }

  static extractVariables(template: string): string[] {
    const variables = new Set<string>();
    
    // Extract simple variables {{variable}}
    const simpleVars = template.match(/{{(?!#|\/)\s*(\w+)\s*}}/g) || [];
    simpleVars.forEach(match => {
      const variable = match.replace(/[{}]/g, '').trim();
      if (!['this', '@index'].includes(variable)) {
        variables.add(variable);
      }
    });
    
    // Extract conditional variables {{#if variable}}
    const conditionalVars = template.match(/{{#if\s+(\w+)}}/g) || [];
    conditionalVars.forEach(match => {
      const variable = match.match(/{{#if\s+(\w+)}}/)?.[1];
      if (variable) variables.add(variable);
    });
    
    // Extract loop variables {{#each variable}}
    const loopVars = template.match(/{{#each\s+(\w+)}}/g) || [];
    loopVars.forEach(match => {
      const variable = match.match(/{{#each\s+(\w+)}}/)?.[1];
      if (variable) variables.add(variable);
    });
    
    return Array.from(variables);
  }
}

export const PROMPT_TEMPLATES = {
  CONTENT_GENERATION: {
    template: `You are a professional content creator. Create {{content_type}} content about {{topic}}.

Requirements:
- Target audience: {{audience}}
- Tone: {{tone}}
- Length: {{length}}
{{#if keywords}}
- Include these keywords: {{keywords}}
{{/if}}
{{#if examples}}
- Use these examples as inspiration:
{{#each examples}}
- {{this}}
{{/each}}
{{/if}}

Please create engaging, high-quality content that meets these requirements.`,
    variables: {
      content_type: 'blog post',
      topic: '',
      audience: 'general',
      tone: 'professional',
      length: 'medium',
      keywords: '',
      examples: []
    }
  },

  DATA_EXTRACTION: {
    template: `Extract structured data from the following text in JSON format.

Text to analyze:
{{input_text}}

Extract the following fields:
{{#each fields}}
- {{name}}: {{description}}
{{/each}}

{{#if format_requirements}}
Format requirements:
{{format_requirements}}
{{/if}}

Return only valid JSON without any additional text.`,
    variables: {
      input_text: '',
      fields: [],
      format_requirements: ''
    }
  },

  CODE_REVIEW: {
    template: `Review the following {{language}} code for:
- Code quality and best practices
- Security vulnerabilities
- Performance issues
- Maintainability
{{#if specific_concerns}}
- Specific concerns: {{specific_concerns}}
{{/if}}

Code to review:
\`\`\`{{language}}
{{code}}
\`\`\`

Provide detailed feedback with specific suggestions for improvement.`,
    variables: {
      language: 'javascript',
      code: '',
      specific_concerns: ''
    }
  },

  CUSTOMER_SUPPORT: {
    template: `You are a helpful customer support agent for {{company_name}}.

Customer inquiry:
{{customer_message}}

{{#if customer_context}}
Customer context:
{{customer_context}}
{{/if}}

{{#if knowledge_base}}
Relevant knowledge base information:
{{knowledge_base}}
{{/if}}

Provide a helpful, professional response that addresses the customer's needs.
Tone: {{tone}}
{{#if escalation_needed}}
If escalation is needed, explain when and how to escalate.
{{/if}}`,
    variables: {
      company_name: '',
      customer_message: '',
      customer_context: '',
      knowledge_base: '',
      tone: 'friendly and professional',
      escalation_needed: false
    }
  }
};