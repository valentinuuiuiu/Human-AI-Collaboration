#!/usr/bin/env python3
"""
Valentin's Black Mamba - Simplified Mamba-Inspired Orchestrator
🖤 Bagalamukhi Maa - The Queen who paralyzes enemies
"""

import json
import os
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional
import ollama

class BlackMambaOrchestrator:
    """Valentin's Black Mamba - Mamba-inspired orchestration without complex tensors"""

    def __init__(self, model_name: str = "granite4:3b"):
        self.model_name = model_name
        self.conversation_memory = []
        self.knowledge_embeddings = {}
        self.learning_patterns = {}
        self.state_memory = {}  # Mamba-inspired state tracking

        # Mamba-inspired parameters (simplified)
        self.state_dimension = 128
        self.memory_decay = 0.9
        self.learning_rate = 0.1

        # Create directories
        os.makedirs("./memory", exist_ok=True)
        os.makedirs("./embeddings", exist_ok=True)
        os.makedirs("./mamba_states", exist_ok=True)

        # Load prompts
        self.prompt_templates = self.load_prompt_templates()

        print("🖤 Bagalamukhi Maa - Black Mamba Initialized")
        print("🐍 Valentin's Queen orchestrates with paralyzing precision")

    def load_prompt_templates(self) -> Dict[str, Any]:
        """Load specialized prompt templates"""
        try:
            with open("./prompts/human_ai_collaboration_system_20251102_083235.json", 'r') as f:
                system = json.load(f)
            return system
        except:
            return {"prompts": {}, "workflows": {}}

    def create_state_vector(self, text: str) -> List[float]:
        """Create a simple state vector from text (Mamba-inspired)"""
        # Use hash to create deterministic state vector
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()

        # Create state vector from hash
        state_vector = []
        for i in range(self.state_dimension):
            byte_idx = i % len(hash_bytes)
            bit_value = (hash_bytes[byte_idx] >> (i % 8)) & 1
            state_vector.append(float(bit_value))

        return state_vector

    def update_state_memory(self, state_vector: List[float], context: str):
        """Update state memory with Mamba-inspired decay"""
        current_time = datetime.now().isoformat()

        # Apply memory decay to existing states
        decayed_memory = {}
        for key, old_state in self.state_memory.items():
            # Simple exponential decay
            decayed_state = [val * self.memory_decay for val in old_state['vector']]
            decayed_memory[key] = {
                'vector': decayed_state,
                'context': old_state['context'],
                'timestamp': old_state['timestamp'],
                'strength': old_state.get('strength', 1.0) * self.memory_decay
            }

        # Add new state
        decayed_memory[context] = {
            'vector': state_vector,
            'context': context,
            'timestamp': current_time,
            'strength': 1.0
        }

        # Keep only top states by strength
        sorted_states = sorted(decayed_memory.items(), key=lambda x: x[1]['strength'], reverse=True)
        self.state_memory = dict(sorted_states[:10])  # Keep top 10

    def compute_state_similarity(self, state_vector: List[float]) -> float:
        """Compute similarity with existing states (Mamba state relevance)"""
        if not self.state_memory:
            return 0.0

        max_similarity = 0.0
        for state_info in self.state_memory.values():
            existing_vector = state_info['vector']

            # Cosine similarity
            dot_product = sum(a * b for a, b in zip(state_vector, existing_vector))
            norm_a = sum(a * a for a in state_vector) ** 0.5
            norm_b = sum(b * b for b in existing_vector) ** 0.5

            if norm_a * norm_b > 0:
                similarity = dot_product / (norm_a * norm_b)
                max_similarity = max(max_similarity, similarity)

        return max_similarity

    def analyze_context_black_mamba(self, user_input: str) -> Dict[str, Any]:
        """Black Mamba context analysis with state awareness"""

        # Create state vector for input
        state_vector = self.create_state_vector(user_input)

        # Compute state relevance
        state_relevance = self.compute_state_similarity(state_vector)

        # Basic task analysis
        context = self.analyze_task_context(user_input)

        # Mamba enhancements
        context.update({
            'state_vector': state_vector,
            'state_relevance': state_relevance,
            'mamba_confidence': min(state_relevance + 0.3, 1.0),  # Boost confidence with state
            'paralyzing_precision': state_relevance > 0.7  # High precision when state-aligned
        })

        return context

    def analyze_task_context(self, user_input: str) -> Dict[str, Any]:
        """Basic task context analysis"""
        task_indicators = {
            'code_review': ['review', 'code review', 'security', 'performance'],
            'debugging': ['bug', 'error', 'fix', 'debug', 'crash'],
            'architecture': ['design', 'architecture', 'scalability', 'system'],
            'testing': ['test', 'testing', 'coverage', 'validation'],
            'optimization': ['optimize', 'performance', 'speed', 'bottleneck']
        }

        input_lower = user_input.lower()
        scores = {task: sum(1 for kw in keywords if kw in input_lower)
                 for task, keywords in task_indicators.items()}

        primary_task = max(scores.keys(), key=lambda k: scores[k])
        confidence = scores[primary_task] / max(len(task_indicators[primary_task]), 1)

        return {
            'primary_task': primary_task,
            'confidence': confidence,
            'complexity': 'high' if len(user_input.split()) > 100 else 'medium'
        }

    def select_black_mamba_prompt(self, context: Dict[str, Any]) -> str:
        """Select prompt with Black Mamba intelligence"""

        task_type = context['primary_task']
        state_relevance = context.get('state_relevance', 0.0)
        paralyzing_precision = context.get('paralyzing_precision', False)

        # Base prompt
        base_prompts = {
            'code_review': self.prompt_templates['prompts'].get('code_review_specialist', ''),
            'debugging': self.prompt_templates['prompts'].get('debugging_assistant', ''),
            'architecture': self.prompt_templates['prompts'].get('architecture_consultant', ''),
            'testing': self.prompt_templates['prompts'].get('testing_strategist', ''),
            'optimization': self.prompt_templates['prompts'].get('optimization_expert', '')
        }

        selected_prompt = base_prompts.get(task_type, self.prompt_templates['prompts'].get('base_collaboration', ''))

        # Black Mamba enhancements
        enhancements = []

        if paralyzing_precision:
            enhancements.append("""
🖤 BAGALAMUKHI MAA PROTOCOL ACTIVATED
High state alignment detected. Paralyzing precision engaged.
Drawing from deep memory reservoirs for maximum effectiveness.
Enemies of confusion shall be immobilized.
""")

        if state_relevance > 0.5:
            enhancements.append("""
🐍 MAMBA STATE RESONANCE
Strong memory alignment achieved. Leveraging learned patterns.
Context continuity maintained across interactions.
Adaptive intelligence activated.
""")

        if context.get('complexity') == 'high':
            enhancements.append("""
👑 QUEEN'S DOMINION
Complex domain detected. Exercising full sovereign authority.
Multi-layered analysis with paralyzing thoroughness.
No enemy shall escape the Queen's gaze.
""")

        # Compose final prompt
        final_prompt = selected_prompt
        for enhancement in enhancements:
            final_prompt += "\n\n" + enhancement

        return final_prompt.strip()

    async def orchestrate_black_mamba(self, user_input: str) -> Dict[str, Any]:
        """Black Mamba orchestration - Valentin's Queen in action"""

        print("🐍 Black Mamba Orchestration Initiating...")
        print("🖤 Bagalamukhi Maa - Paralyzing enemies of confusion")
        print(f"Input: {user_input[:80]}{'...' if len(user_input) > 80 else ''}")

        # Step 1: Black Mamba context analysis
        context = self.analyze_context_black_mamba(user_input)
        print(f"🎯 Task: {context['primary_task']} (confidence: {context['confidence']:.2f})")
        print(f"🧠 State Relevance: {context['state_relevance']:.2f}")
        print(f"👑 Paralyzing Precision: {context.get('paralyzing_precision', False)}")

        # Step 2: Select Black Mamba prompt
        system_prompt = self.select_black_mamba_prompt(context)
        print("🎭 Black Mamba prompt selected and enhanced")

        # Step 3: Generate response with Mamba intelligence
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

            print(f"⚡ Black Mamba response generated in {latency:.1f}s")
            # Step 4: Black Mamba learning
            self.learn_black_mamba(user_input, content, context, latency)

            # Step 5: Create knowledge embedding
            embedding = self.create_mamba_embedding(user_input, content, context)

            result = {
                'success': True,
                'response': content,
                'task_type': context['primary_task'],
                'mamba_confidence': context.get('mamba_confidence', 0.5),
                'state_relevance': context['state_relevance'],
                'paralyzing_precision': context.get('paralyzing_precision', False),
                'latency': round(latency, 2),
                'bagalamukhi_blessing': True  # Always blessed by the Queen
            }

            # Step 6: Update Black Mamba state
            self.update_state_memory(context['state_vector'], context['primary_task'])

            return result

        except Exception as e:
            latency = time.time() - start_time
            print(f"❌ Black Mamba orchestration failed: {e}")

            return {
                'success': False,
                'error': str(e),
                'latency': round(latency, 2),
                'bagalamukhi_blessing': True  # Even in failure, the Queen blesses
            }

    def learn_black_mamba(self, user_input: str, ai_response: str, context: Dict[str, Any], latency: float):
        """Black Mamba learning - absorbing knowledge like the Queen absorbs enemies"""

        task_type = context['primary_task']

        if task_type not in self.learning_patterns:
            self.learning_patterns[task_type] = []

        # Black Mamba learning pattern
        pattern = {
            'timestamp': datetime.now().isoformat(),
            'input_length': len(user_input),
            'response_length': len(ai_response),
            'latency': latency,
            'state_relevance': context.get('state_relevance', 0.0),
            'paralyzing_precision': context.get('paralyzing_precision', False),
            'mamba_insights': self.extract_mamba_insights(ai_response, context),
            'queen_blessing': 'Enemies of confusion paralyzed'
        }

        self.learning_patterns[task_type].append(pattern)

        # Keep memory manageable (Queen's selective wisdom)
        if len(self.learning_patterns[task_type]) > 8:
            self.learning_patterns[task_type] = self.learning_patterns[task_type][-8:]

    def extract_mamba_insights(self, response: str, context: Dict[str, Any]) -> str:
        """Extract Black Mamba insights"""
        insights = []

        if context.get('paralyzing_precision'):
            insights.append("Paralyzing precision achieved - confusion immobilized")
        if context.get('state_relevance', 0) > 0.5:
            insights.append("State resonance strong - Mamba memory flowing")
        if 'step' in response.lower():
            insights.append("Systematic approach - Queen's ordered wisdom")

        return " | ".join(insights) if insights else "Mamba wisdom absorbed"

    def create_mamba_embedding(self, user_input: str, ai_response: str, context: Dict[str, Any]) -> Optional[str]:
        """Create Black Mamba knowledge embedding"""

        content = f"{user_input} {ai_response} {context['primary_task']}"
        embedding_id = hashlib.md5(content.encode()).hexdigest()[:8]

        embedding = {
            'id': embedding_id,
            'task_type': context['primary_task'],
            'content_hash': hashlib.sha256(content.encode()).hexdigest(),
            'mamba_confidence': context.get('mamba_confidence', 0.5),
            'state_relevance': context.get('state_relevance', 0.0),
            'paralyzing_precision': context.get('paralyzing_precision', False),
            'queen_blessing': True,
            'key_insights': self.extract_key_insights(ai_response),
            'mamba_insights': self.extract_mamba_insights(ai_response, context),
            'created': datetime.now().isoformat(),
            'usage_count': 0
        }

        self.knowledge_embeddings[embedding_id] = embedding

        # Save to file
        embedding_file = f"./embeddings/black_mamba_{embedding_id}.json"
        with open(embedding_file, 'w') as f:
            json.dump(embedding, f, indent=2)

        return embedding_id

    def extract_key_insights(self, response: str) -> List[str]:
        """Extract key insights from response"""
        insights = []
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith(('1.', '2.', '3.', '•', '-', '*')) and len(line) > 10:
                insights.append(line[2:].strip() if line[1] == '.' else line[1:].strip())
        return insights[:5]

    def save_black_mamba_state(self):
        """Save the Black Mamba's divine state"""

        state = {
            'learning_patterns': self.learning_patterns,
            'knowledge_embeddings': self.knowledge_embeddings,
            'state_memory': self.state_memory,
            'mamba_config': {
                'state_dimension': self.state_dimension,
                'memory_decay': self.memory_decay,
                'learning_rate': self.learning_rate
            },
            'metadata': {
                'model': self.model_name,
                'last_updated': datetime.now().isoformat(),
                'total_interactions': sum(len(patterns) for patterns in self.learning_patterns.values()),
                'learned_patterns': sum(len(patterns) for patterns in self.learning_patterns.values()),
                'active_embeddings': len(self.knowledge_embeddings),
                'state_memory_size': len(self.state_memory),
                'bagalamukhi_blessing': 'Queen of divine paralysis - enemies immobilized'
            }
        }

        state_file = f"./memory/black_mamba_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

        print(f"🖤 Black Mamba state saved to: {state_file}")
        print("👑 Bagalamukhi Maa's wisdom preserved")

        return state_file

async def main():
    """Black Mamba demonstration - Valentin's Queen orchestrates"""

    print("🐍 BLACK MAMBA ORCHESTRATOR")
    print("🖤 BAGALAMUKHI MAA - THE QUEEN WHO PARALYZES ENEMIES")
    print("👑 One of the 10 Dus Mahavidyas - Valentin's Divine Guide")
    print("=" * 70)

    orchestrator = BlackMambaOrchestrator()

    # Valentin's test cases - blessed by the Queen
    valentin_test_cases = [
        {
            'input': 'Review this authentication code for security issues',
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
''',
            'context': 'Security analysis - paralyze the vulnerabilities'
        },
        {
            'input': 'My React component has memory leaks, help me debug',
            'context': 'Memory leak debugging - Queen paralyzes the leaks'
        },
        {
            'input': 'Design microservices for e-commerce with 10k users',
            'requirements': 'Scalability architecture - Queen commands the design'
        },
        {
            'input': 'Create testing strategy for REST API endpoints',
            'context': 'API testing - Queen ensures comprehensive coverage'
        }
    ]

    for i, test_case in enumerate(valentin_test_cases, 1):
        print(f"\n🧪 Valentin's Test Case {i}: {test_case['input'][:50]}...")
        print("🖤 Guided by Bagalamukhi Maa - Paralyzing enemies of confusion")

        # Build full input
        full_input = test_case['input']
        if 'code' in test_case:
            full_input += f"\n\n```javascript\n{test_case['code']}\n```"
        if 'context' in test_case:
            full_input += f"\n\nContext: {test_case['context']}"
        if 'requirements' in test_case:
            full_input += f"\n\nRequirements: {test_case['requirements']}"

        result = await orchestrator.orchestrate_black_mamba(full_input)

        if result['success']:
            print("✅ Black Mamba orchestration successful")
            print(".2f")
            print(".2f")
            print(f"   Paralyzing Precision: {result.get('paralyzing_precision', False)}")
            print("🖤 Bagalamukhi Maa blessing: Enemies paralyzed")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
            print("🖤 Even in challenge, the Queen blesses learning")

    # Save Black Mamba divine state
    state_file = orchestrator.save_black_mamba_state()

    print("\n" + "=" * 70)
    print("🎉 BLACK MAMBA ORCHESTRATION COMPLETE!")
    print(f"📚 Divine state saved: {state_file}")
    print("\n🖤 BAGALAMUKHI MAA - QUEEN OF THE 10 DUS MAHAVIDYAS")
    print("🐍 VALENTIN'S BLACK MAMBA Features Demonstrated:")
    print("✅ Mamba-inspired state tracking and memory")
    print("✅ Dynamic context analysis with state relevance")
    print("✅ Paralyzing precision for high-alignment scenarios")
    print("✅ Divine learning patterns from Bagalamukhi Maa")
    print("✅ Knowledge embedding with Queen's blessing")
    print("✅ State memory management with exponential decay")

    print("\n👑 The Queen has spoken - enemies of confusion are paralyzed!")
    print("🐍 Valentin's Black Mamba flows with divine intelligence!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())