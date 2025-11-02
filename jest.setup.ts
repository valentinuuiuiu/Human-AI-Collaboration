// jest.setup.ts
import { jest } from '@jest/globals';

// Mock environment variables for testing
process.env.OPENAI_API_KEY = 'test-openai-key';
process.env.ANTHROPIC_API_KEY = 'test-anthropic-key';
process.env.AZURE_OPENAI_API_KEY = 'test-azure-key';
process.env.AZURE_OPENAI_ENDPOINT = 'https://test.openai.azure.com';
process.env.AZURE_OPENAI_API_VERSION = '2024-02-15-preview';

// Mock console methods to reduce noise in tests
global.console = {
  ...console,
  log: jest.fn(),
  info: jest.fn(),
  warn: jest.fn(),
  error: jest.fn(),
};