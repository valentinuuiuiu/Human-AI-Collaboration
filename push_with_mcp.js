const { spawn } = require('child_process');

// MCP protocol messages
const initializeMsg = {
  jsonrpc: '2.0',
  id: 1,
  method: 'initialize',
  params: {
    protocolVersion: '2024-11-05',
    capabilities: {},
    clientInfo: {
      name: 'git-push-client',
      version: '1.0.0'
    }
  }
};

const toolsListMsg = {
  jsonrpc: '2.0',
  id: 2,
  method: 'tools/list'
};

const pushMsg = {
  jsonrpc: '2.0',
  id: 3,
  method: 'tools/call',
  params: {
    name: 'git_push',
    arguments: {
      remote: 'origin',
      branch: 'main'
    }
  }
};

// Start the MCP server
const server = spawn('npx', ['@cyanheads/git-mcp-server'], {
  stdio: ['pipe', 'pipe', 'pipe'],
  cwd: process.cwd()
});

let messageQueue = [initializeMsg, toolsListMsg, pushMsg];
let currentMsgIndex = 0;

server.stdout.on('data', (data) => {
  console.log('Server output:', data.toString());
  
  // Send next message if available
  if (currentMsgIndex < messageQueue.length) {
    const msg = messageQueue[currentMsgIndex++];
    server.stdin.write(JSON.stringify(msg) + '\n');
  }
});

server.stderr.on('data', (data) => {
  console.error('Server error:', data.toString());
});

server.on('close', (code) => {
  console.log(`Server exited with code ${code}`);
});

// Send first message
setTimeout(() => {
  if (messageQueue.length > 0) {
    const msg = messageQueue[currentMsgIndex++];
    server.stdin.write(JSON.stringify(msg) + '\n');
  }
}, 1000);
