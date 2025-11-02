#!/usr/bin/env python3
"""
Advanced Human-AI Collaboration Orchestrator
Dynamic prompt selection, learning, and knowledge condensation
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import ollama
import hashlib

class AdvancedCollaborationOrchestrator:
    def __init__(self, model_name: str = "granite4:3b"):
        self.model_name = model_name
        self.conversation_memory = []
        self.knowledge_embeddings = {}
        self.learning_patterns = {}
        self.prompt_templates = self.load_prompt_templates()

        # Create memory directories
        os.makedirs("./memory", exist_ok=True)
        os.makedirs("./embeddings", exist_ok=True)

    def load_prompt_templates(self) -> Dict[str, Any]:
        """Load all specialized prompt templates"""
        try:
            with open("./prompts/human_ai_collaboration_system_20251102_083235.json", 'r') as f:
                system = json.load(f)
            return system
        except:
            return {"prompts": {}, "workflows": {}}

    def analyze_task_context(self, user_input: str) -> Dict[str, Any]:
        """Analyze user input to determine task type and context"""

        # Simple keyword-based analysis (could be enhanced with embeddings)
        task_indicators = {
            'code_review': ['review', 'code review', 'pr', 'pull request', 'security', 'performance'],
            'debugging': ['bug', 'error', 'fix', 'debug', 'crash', 'memory leak', 'issue'],
            'architecture': ['design', 'architecture', 'microservices', 'scalability', 'system'],
            'testing': ['test', 'testing', 'unit test', 'integration', 'coverage'],
            'optimization': ['optimize', 'performance', 'speed', 'bottleneck', 'slow'],
            'general': ['help', 'how', 'what', 'explain']
        }

        input_lower = user_input.lower()
        scores = {}

        for task_type, keywords in task_indicators.items():
            score = sum(1 for keyword in keywords if keyword in input_lower)
            scores[task_type] = score

        # Determine primary task type
        primary_task = max(scores.keys(), key=lambda k: scores[k])
        confidence = scores[primary_task] / max(len(task_indicators[primary_task]), 1)

        # Extract code if present
        code_blocks = []
        if '```' in user_input:
            parts = user_input.split('```')
            for i in range(1, len(parts), 2):
                if i < len(parts):
                    code_blocks.append(parts[i])

        return {
            'primary_task': primary_task,
            'confidence': confidence,
            'code_detected': len(code_blocks) > 0,
            'code_blocks': code_blocks,
            'complexity': self.assess_complexity(user_input),
            'keywords': [k for k in task_indicators[primary_task] if k in input_lower]
        }

    def assess_complexity(self, text: str) -> str:
        """Assess task complexity"""
        word_count = len(text.split())
        code_indicators = ['function', 'class', 'import', 'def ', 'const ', 'let ']

        if word_count > 200 or sum(1 for ind in code_indicators if ind in text) > 3:
            return 'high'
        elif word_count > 50 or any(ind in text for ind in code_indicators):
            return 'medium'
        else:
            return 'low'

    def select_dynamic_prompt(self, context: Dict[str, Any]) -> str:
        """Dynamically select and compose the optimal prompt"""

        task_type = context['primary_task']
        complexity = context['complexity']

        # Base prompt selection
        base_prompts = {
            'code_review': self.prompt_templates['prompts'].get('code_review_specialist', ''),
            'debugging': self.prompt_templates['prompts'].get('debugging_assistant', ''),
            'architecture': self.prompt_templates['prompts'].get('architecture_consultant', ''),
            'testing': self.prompt_templates['prompts'].get('testing_strategist', ''),
            'optimization': self.prompt_templates['prompts'].get('optimization_expert', ''),
            'general': self.prompt_templates['prompts'].get('base_collaboration', '')
        }

        selected_prompt = base_prompts.get(task_type, base_prompts['general'])

        # Enhance based on complexity and context
        enhancements = []

        if complexity == 'high':
            enhancements.append("""
COMPLEX TASK PROTOCOL:
- Break down the problem into smaller, manageable components
- Provide step-by-step analysis and solutions
- Consider edge cases and potential failure modes
- Suggest iterative implementation approach
- Include validation and testing strategies
""")

        if context['code_detected']:
            enhancements.append("""
CODE ANALYSIS MODE:
- Analyze code structure and patterns
- Identify potential issues and improvements
- Suggest specific code changes with examples
- Consider performance, security, and maintainability
- Provide runnable code snippets when appropriate
""")

        # Add learning context from previous interactions
        learning_context = self.get_learning_context(task_type)
        if learning_context:
            enhancements.append(f"""
LEARNING CONTEXT:
{learning_context}
""")

        # Compose final prompt
        final_prompt = selected_prompt
        for enhancement in enhancements:
            final_prompt += "\n\n" + enhancement

        return final_prompt.strip()

    def get_learning_context(self, task_type: str) -> str:
        """Retrieve relevant learning context from previous interactions"""

        if task_type not in self.learning_patterns:
            return ""

        patterns = self.learning_patterns[task_type]
        if not patterns:
            return ""

        # Get most relevant patterns (simplified - could use embeddings)
        recent_patterns = patterns[-3:]  # Last 3 interactions

        context_parts = []
        for pattern in recent_patterns:
            if 'successful_approach' in pattern:
                context_parts.append(f"- Previous success: {pattern['successful_approach']}")
            if 'lessons_learned' in pattern:
                context_parts.append(f"- Key lesson: {pattern['lessons_learned']}")

        if context_parts:
            return "RELEVANT LEARNING FROM PREVIOUS INTERACTIONS:\n" + "\n".join(context_parts)

        return ""

    async def orchestrate_collaboration(self, user_input: str) -> Dict[str, Any]:
        """Main orchestration function"""

        print("🎭 Advanced Collaboration Orchestration Starting...")
        print(f"Input: {user_input[:100]}{'...' if len(user_input) > 100 else ''}")

        # Step 1: Analyze context
        context = self.analyze_task_context(user_input)
        print(f"📊 Context Analysis: {context['primary_task']} (confidence: {context['confidence']:.2f})")

        # Step 2: Select dynamic prompt
        system_prompt = self.select_dynamic_prompt(context)
        print("🎯 Dynamic prompt selected and enhanced")

        # Step 3: Generate response with learning
        start_time = time.time()

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_input}
                ],
                options={'temperature': 0.7, 'num_predict': 1000}
            )

            latency = time.time() - start_time
            content = response['message']['content']

            print(f"⚡ Response generated in {latency:.1f}s")
            # Step 4: Learn from interaction
            self.learn_from_interaction(user_input, content, context, latency)

            # Step 5: Create knowledge embedding
            embedding = self.create_knowledge_embedding(user_input, content, context)

            result = {
                'success': True,
                'response': content,
                'task_type': context['primary_task'],
                'confidence': context['confidence'],
                'complexity': context['complexity'],
                'latency': round(latency, 2),
                'learning_applied': bool(self.get_learning_context(context['primary_task'])),
                'embedding_created': embedding is not None,
                'timestamp': datetime.now().isoformat()
            }

            # Step 6: Update conversation memory
            self.update_memory(user_input, content, result)

            return result

        except Exception as e:
            latency = time.time() - start_time
            print(f"❌ Orchestration failed: {e}")

            return {
                'success': False,
                'error': str(e),
                'latency': round(latency, 2),
                'timestamp': datetime.now().isoformat()
            }

    def learn_from_interaction(self, user_input: str, ai_response: str, context: Dict[str, Any], latency: float):
        """Learn patterns from successful interactions"""

        task_type = context['primary_task']

        if task_type not in self.learning_patterns:
            self.learning_patterns[task_type] = []

        # Extract learning patterns (simplified)
        pattern = {
            'timestamp': datetime.now().isoformat(),
            'input_length': len(user_input),
            'response_length': len(ai_response),
            'latency': latency,
            'complexity': context['complexity'],
            'successful_approach': self.extract_successful_approach(ai_response),
            'lessons_learned': self.extract_lessons(ai_response)
        }

        self.learning_patterns[task_type].append(pattern)

        # Keep only recent patterns (memory management)
        if len(self.learning_patterns[task_type]) > 10:
            self.learning_patterns[task_type] = self.learning_patterns[task_type][-10:]

    def extract_successful_approach(self, response: str) -> str:
        """Extract successful approaches from AI response"""
        # Simple pattern extraction
        if 'step-by-step' in response.lower():
            return "Used systematic step-by-step approach"
        elif 'consider' in response.lower() and 'alternative' in response.lower():
            return "Provided multiple solution alternatives"
        elif 'security' in response.lower() and 'vulnerability' in response.lower():
            return "Focused on security analysis"
        else:
            return "Provided structured guidance"

    def extract_lessons(self, response: str) -> str:
        """Extract lessons learned from response"""
        # Look for improvement suggestions
        if 'improve' in response.lower():
            return "Emphasized iterative improvement"
        elif 'test' in response.lower():
            return "Stressed importance of testing"
        elif 'maintain' in response.lower():
            return "Focused on long-term maintainability"
        else:
            return "Applied domain-specific best practices"

    def create_knowledge_embedding(self, user_input: str, ai_response: str, context: Dict[str, Any]) -> Optional[str]:
        """Create a knowledge embedding for future learning"""

        # Simple hash-based embedding (could be enhanced with real embeddings)
        content = f"{user_input} {ai_response} {context['primary_task']}"
        embedding_id = hashlib.md5(content.encode()).hexdigest()[:8]

        embedding = {
            'id': embedding_id,
            'task_type': context['primary_task'],
            'content_hash': hashlib.sha256(content.encode()).hexdigest(),
            'key_insights': self.extract_key_insights(ai_response),
            'created': datetime.now().isoformat(),
            'usage_count': 0
        }

        self.knowledge_embeddings[embedding_id] = embedding

        # Save to file
        embedding_file = f"./embeddings/{embedding_id}.json"
        with open(embedding_file, 'w') as f:
            json.dump(embedding, f, indent=2)

        return embedding_id

    def extract_key_insights(self, response: str) -> List[str]:
        """Extract key insights from AI response"""
        insights = []

        # Look for numbered lists, bullet points, or key recommendations
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith(('1.', '2.', '3.', '•', '-', '*')) and len(line) > 10:
                insights.append(line[2:].strip() if line[1] == '.' else line[1:].strip())

        return insights[:5]  # Limit to top 5 insights

    def update_memory(self, user_input: str, ai_response: str, result: Dict[str, Any]):
        """Update conversation memory"""

        memory_entry = {
            'timestamp': result['timestamp'],
            'user_input': user_input,
            'ai_response': ai_response,
            'task_type': result.get('task_type'),
            'success': result['success'],
            'latency': result.get('latency'),
            'learning_applied': result.get('learning_applied', False)
        }

        self.conversation_memory.append(memory_entry)

        # Keep memory manageable
        if len(self.conversation_memory) > 50:
            self.conversation_memory = self.conversation_memory[-50:]

    def save_orchestrator_state(self):
        """Save the orchestrator's learning state"""

        state = {
            'learning_patterns': self.learning_patterns,
            'knowledge_embeddings': self.knowledge_embeddings,
            'conversation_memory': self.conversation_memory[-20:],  # Last 20 conversations
            'metadata': {
                'model': self.model_name,
                'last_updated': datetime.now().isoformat(),
                'total_interactions': len(self.conversation_memory),
                'learned_patterns': sum(len(patterns) for patterns in self.learning_patterns.values())
            }
        }

        state_file = f"./memory/orchestrator_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

        print(f"💾 Orchestrator state saved to: {state_file}")
        return state_file

async def main():
    """Demonstrate the advanced orchestration system"""

    print("🚀 Advanced Human-AI Collaboration Orchestrator")
    print("=" * 60)

    orchestrator = AdvancedCollaborationOrchestrator()

    # Example interactions to demonstrate learning
    test_cases = [
        {
            'input': 'Please review this authentication code for security issues',
            'code': '''
function authenticateUser(req, res, next) {
    const token = req.headers.authorization?.split(' ')[1];
    if (!token) return res.status(401).json({ error: 'No token provided' });
    try {
        const decoded = jwt.verify(token, process.env.JWT_SECRET);
        req.user = decoded;
        next();
    } catch (error) {
        return res.status(401).json({ error: 'Invalid token' });
    }
}
'''
        },
        {
            'input': 'My React app is slow and has memory leaks. What should I check?',
            'context': 'React component with useEffect and event listeners'
        },
        {
            'input': 'Design a microservices architecture for an e-commerce site',
            'requirements': '10k concurrent users, real-time inventory, payment processing'
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test Case {i}: {test_case['input'][:50]}...")

        # Combine input with any additional context
        full_input = test_case['input']
        if 'code' in test_case:
            full_input += f"\n\n```javascript\n{test_case['code']}\n```"
        if 'context' in test_case:
            full_input += f"\n\nContext: {test_case['context']}"
        if 'requirements' in test_case:
            full_input += f"\n\nRequirements: {test_case['requirements']}"

        result = await orchestrator.orchestrate_collaboration(full_input)

        if result['success']:
            print("✅ Orchestration successful")
            print(f"   Learned: {result.get('learning_applied', False)}")
            print(f"   Embedding: {result.get('embedding_created', False)}")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")

    # Save learning state
    state_file = orchestrator.save_orchestrator_state()

    print("\n" + "=" * 60)
    print("🎉 Advanced Orchestration Demo Complete!")
    print(f"📚 Learning state saved: {state_file}")
    print("\nKey Features Demonstrated:")
    print("✅ Dynamic prompt selection based on context")
    print("✅ Learning from interaction patterns")
    print("✅ Knowledge embedding creation")
    print("✅ Memory management and state persistence")
    print("✅ Complexity assessment and adaptive responses")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())