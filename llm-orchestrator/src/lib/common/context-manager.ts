import { v4 as uuidv4 } from 'uuid';
import { WorkflowContext, LLMMessage, LLMResponse } from './types';

export class ContextManager {
  private contexts: Map<string, WorkflowContext> = new Map();
  private maxContexts: number = 1000;
  private maxMessagesPerContext: number = 100;

  createContext(initialVariables: Record<string, any> = {}): string {
    const id = uuidv4();
    const context: WorkflowContext = {
      id,
      messages: [],
      variables: { ...initialVariables },
      results: [],
      metadata: {
        created: new Date().toISOString(),
        lastUpdated: new Date().toISOString()
      }
    };

    this.contexts.set(id, context);
    this.cleanupOldContexts();
    return id;
  }

  getContext(id: string): WorkflowContext | undefined {
    return this.contexts.get(id);
  }

  updateContext(id: string, updates: Partial<WorkflowContext>): void {
    const context = this.contexts.get(id);
    if (!context) {
      throw new Error(`Context ${id} not found`);
    }

    Object.assign(context, updates, {
      metadata: {
        ...context.metadata,
        lastUpdated: new Date().toISOString()
      }
    });

    this.contexts.set(id, context);
  }

  addMessage(contextId: string, message: LLMMessage): void {
    const context = this.getContext(contextId);
    if (!context) {
      throw new Error(`Context ${contextId} not found`);
    }

    context.messages.push(message);
    
    // Trim messages if exceeding limit
    if (context.messages.length > this.maxMessagesPerContext) {
      // Keep system messages and recent messages
      const systemMessages = context.messages.filter(msg => msg.role === 'system');
      const recentMessages = context.messages
        .filter(msg => msg.role !== 'system')
        .slice(-this.maxMessagesPerContext + systemMessages.length);
      
      context.messages = [...systemMessages, ...recentMessages];
    }

    this.updateContext(contextId, { messages: context.messages });
  }

  addResult(contextId: string, result: LLMResponse): void {
    const context = this.getContext(contextId);
    if (!context) {
      throw new Error(`Context ${contextId} not found`);
    }

    context.results.push(result);
    this.updateContext(contextId, { results: context.results });
  }

  setVariable(contextId: string, key: string, value: any): void {
    const context = this.getContext(contextId);
    if (!context) {
      throw new Error(`Context ${contextId} not found`);
    }

    context.variables[key] = value;
    this.updateContext(contextId, { variables: context.variables });
  }

  getVariable(contextId: string, key: string): any {
    const context = this.getContext(contextId);
    if (!context) {
      throw new Error(`Context ${contextId} not found`);
    }

    return context.variables[key];
  }

  getMessages(contextId: string, role?: string): LLMMessage[] {
    const context = this.getContext(contextId);
    if (!context) {
      throw new Error(`Context ${contextId} not found`);
    }

    if (role) {
      return context.messages.filter(msg => msg.role === role);
    }
    return context.messages;
  }

  getLastResult(contextId: string): LLMResponse | undefined {
    const context = this.getContext(contextId);
    if (!context) {
      throw new Error(`Context ${contextId} not found`);
    }

    return context.results[context.results.length - 1];
  }

  summarizeContext(contextId: string): string {
    const context = this.getContext(contextId);
    if (!context) {
      throw new Error(`Context ${contextId} not found`);
    }

    const messageCount = context.messages.length;
    const resultCount = context.results.length;
    const variableCount = Object.keys(context.variables).length;
    
    return `Context ${contextId}: ${messageCount} messages, ${resultCount} results, ${variableCount} variables`;
  }

  exportContext(contextId: string): WorkflowContext {
    const context = this.getContext(contextId);
    if (!context) {
      throw new Error(`Context ${contextId} not found`);
    }

    return JSON.parse(JSON.stringify(context));
  }

  importContext(contextData: WorkflowContext): string {
    this.contexts.set(contextData.id, contextData);
    return contextData.id;
  }

  deleteContext(contextId: string): boolean {
    return this.contexts.delete(contextId);
  }

  listContexts(): string[] {
    return Array.from(this.contexts.keys());
  }

  private cleanupOldContexts(): void {
    if (this.contexts.size <= this.maxContexts) {
      return;
    }

    // Sort contexts by last updated time and remove oldest
    const sortedContexts = Array.from(this.contexts.entries())
      .sort((a, b) => {
        const timeA = new Date(a[1].metadata.lastUpdated || 0).getTime();
        const timeB = new Date(b[1].metadata.lastUpdated || 0).getTime();
        return timeA - timeB;
      });

    const toRemove = sortedContexts.slice(0, this.contexts.size - this.maxContexts + 1);
    toRemove.forEach(([id]) => this.contexts.delete(id));
  }

  getContextStats(): {
    totalContexts: number;
    totalMessages: number;
    totalResults: number;
    averageMessagesPerContext: number;
  } {
    const totalContexts = this.contexts.size;
    let totalMessages = 0;
    let totalResults = 0;

    for (const context of this.contexts.values()) {
      totalMessages += context.messages.length;
      totalResults += context.results.length;
    }

    return {
      totalContexts,
      totalMessages,
      totalResults,
      averageMessagesPerContext: totalContexts > 0 ? totalMessages / totalContexts : 0
    };
  }
}