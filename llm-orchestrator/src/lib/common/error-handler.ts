export class LLMError extends Error {
  constructor(
    message: string,
    public code: string,
    public provider?: string,
    public retryable: boolean = false,
    public metadata?: Record<string, any>
  ) {
    super(message);
    this.name = 'LLMError';
  }
}

export class APIError extends LLMError {
  constructor(
    message: string,
    public statusCode: number,
    provider?: string,
    retryable: boolean = true
  ) {
    super(message, 'API_ERROR', provider, retryable, { statusCode });
    this.name = 'APIError';
  }
}

export class RateLimitError extends LLMError {
  constructor(
    message: string,
    public resetTime?: Date,
    provider?: string
  ) {
    super(message, 'RATE_LIMIT', provider, true, { resetTime });
    this.name = 'RateLimitError';
  }
}

export class ValidationError extends LLMError {
  constructor(
    message: string,
    public field?: string,
    public value?: any
  ) {
    super(message, 'VALIDATION_ERROR', undefined, false, { field, value });
    this.name = 'ValidationError';
  }
}

export class TimeoutError extends LLMError {
  constructor(
    message: string,
    public timeoutMs: number,
    provider?: string
  ) {
    super(message, 'TIMEOUT', provider, true, { timeoutMs });
    this.name = 'TimeoutError';
  }
}

export class ConfigurationError extends LLMError {
  constructor(
    message: string,
    public configKey?: string
  ) {
    super(message, 'CONFIG_ERROR', undefined, false, { configKey });
    this.name = 'ConfigurationError';
  }
}

export class NetworkError extends LLMError {
  constructor(
    message: string,
    public originalError?: Error,
    provider?: string
  ) {
    super(message, 'NETWORK_ERROR', provider, true, { originalError: originalError?.message });
    this.name = 'NetworkError';
  }
}

export interface ErrorRecoveryStrategy {
  type: 'retry' | 'fallback' | 'circuit_breaker' | 'graceful_degradation';
  config: Record<string, any>;
}

export interface ErrorHandlerConfig {
  maxRetries: number;
  retryDelay: number;
  backoffMultiplier: number;
  timeoutMs: number;
  circuitBreakerThreshold: number;
  recoveryStrategies: ErrorRecoveryStrategy[];
}

export class ErrorHandler {
  private circuitBreakerState: Map<string, { failures: number; lastFailure: Date; state: 'closed' | 'open' | 'half-open' }> = new Map();

  constructor(private config: ErrorHandlerConfig) {}

  async executeWithRetry<T>(
    operation: () => Promise<T>,
    context: string,
    provider?: string
  ): Promise<T> {
    let lastError: Error | undefined;
    let attempt = 0;

    while (attempt < this.config.maxRetries) {
      try {
        // Check circuit breaker
        if (this.isCircuitBreakerOpen(provider)) {
          throw new LLMError(
            `Circuit breaker is open for provider ${provider}`,
            'CIRCUIT_BREAKER_OPEN',
            provider,
            false
          );
        }

        const result = await this.executeWithTimeout(operation, this.config.timeoutMs);
        this.recordSuccess(provider);
        return result;

      } catch (error) {
        lastError = error as Error;
        attempt++;

        // Record failure for circuit breaker
        this.recordFailure(provider);

        // Check if error is retryable
        if (!this.isRetryableError(error) || attempt >= this.config.maxRetries) {
          break;
        }

        // Calculate delay with exponential backoff
        const delay = this.calculateDelay(attempt);
        await this.delay(delay);
      }
    }

    throw this.enhanceError(lastError || new Error('Unknown error'), context, attempt);
  }

  private async executeWithTimeout<T>(operation: () => Promise<T>, timeoutMs: number): Promise<T> {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new TimeoutError(`Operation timed out after ${timeoutMs}ms`, timeoutMs));
      }, timeoutMs);

      operation()
        .then((result) => {
          clearTimeout(timeout);
          resolve(result);
        })
        .catch((error) => {
          clearTimeout(timeout);
          reject(error);
        });
    });
  }

  private isRetryableError(error: any): boolean {
    if (error instanceof LLMError) {
      return error.retryable;
    }

    // Check for common retryable HTTP status codes
    if (error.statusCode) {
      return [408, 429, 500, 502, 503, 504].includes(error.statusCode);
    }

    // Check for network-related errors
    if (error.code) {
      return ['ECONNRESET', 'ETIMEDOUT', 'ENOTFOUND', 'ECONNREFUSED'].includes(error.code);
    }

    return false;
  }

  private calculateDelay(attempt: number): number {
    const baseDelay = this.config.retryDelay;
    const multiplier = Math.pow(this.config.backoffMultiplier, attempt - 1);
    const jitter = Math.random() * 0.1 * baseDelay; // Add 10% jitter
    return Math.min(baseDelay * multiplier + jitter, 30000); // Cap at 30 seconds
  }

  private async delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  private isCircuitBreakerOpen(provider?: string): boolean {
    if (!provider) return false;

    const state = this.circuitBreakerState.get(provider);
    if (!state) return false;

    if (state.state === 'open') {
      // Check if we should transition to half-open
      const timeSinceLastFailure = Date.now() - state.lastFailure.getTime();
      if (timeSinceLastFailure > 60000) { // 1 minute timeout
        state.state = 'half-open';
        return false;
      }
      return true;
    }

    return false;
  }

  private recordSuccess(provider?: string): void {
    if (!provider) return;

    const state = this.circuitBreakerState.get(provider);
    if (state && state.state === 'half-open') {
      state.failures = 0;
      state.state = 'closed';
    }
  }

  private recordFailure(provider?: string): void {
    if (!provider) return;

    const state = this.circuitBreakerState.get(provider) || {
      failures: 0,
      lastFailure: new Date(),
      state: 'closed' as const
    };

    state.failures++;
    state.lastFailure = new Date();

    if (state.failures >= this.config.circuitBreakerThreshold) {
      state.state = 'open';
    }

    this.circuitBreakerState.set(provider, state);
  }

  private enhanceError(error: Error, context: string, attempts: number): Error {
    const enhancedMessage = `${error.message} (Context: ${context}, Attempts: ${attempts})`;

    if (error instanceof LLMError) {
      return new LLMError(
        enhancedMessage,
        error.code,
        error.provider,
        error.retryable,
        { ...error.metadata, context, attempts }
      );
    }

    return new LLMError(
      enhancedMessage,
      'UNKNOWN_ERROR',
      undefined,
      false,
      { originalError: error.message, context, attempts }
    );
  }

  async executeWithFallback<T>(
    operations: (() => Promise<T>)[],
    context: string
  ): Promise<T> {
    for (let i = 0; i < operations.length; i++) {
      try {
        return await operations[i]();
      } catch (error) {
        if (i === operations.length - 1) {
          // Last operation failed
          throw error;
        }
        // Log fallback attempt and continue
        console.warn(`Fallback attempt ${i + 1} failed:`, error);
      }
    }

    throw new LLMError('All fallback operations failed', 'FALLBACK_FAILED');
  }
}

// Default error handler configuration
export const defaultErrorHandlerConfig: ErrorHandlerConfig = {
  maxRetries: 3,
  retryDelay: 1000,
  backoffMultiplier: 2,
  timeoutMs: 30000,
  circuitBreakerThreshold: 5,
  recoveryStrategies: [
    {
      type: 'retry',
      config: { maxRetries: 3 }
    },
    {
      type: 'circuit_breaker',
      config: { threshold: 5, timeout: 60000 }
    }
  ]
};