#!/usr/bin/env python3
"""
Human-AI Collaboration Benchmarking Framework
Tests and measures collaboration quality across different models and scenarios
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import statistics
import ollama

class CollaborationBenchmark:
    def __init__(self):
        self.models = {
            'granite4:3b': {'provider': 'ollama', 'model': 'granite4:3b'},
            'llama3.2:3b': {'provider': 'ollama', 'model': 'llama3.2:3b'},
            # Add OpenAI if API key available
            # 'gpt-4': {'provider': 'openai', 'model': 'gpt-4', 'api_key': os.getenv('OPENAI_API_KEY')}
        }

        self.test_scenarios = self.load_test_scenarios()
        self.results = {}
        self.metrics = {}

        # Create results directory
        os.makedirs("./benchmarks", exist_ok=True)

    def load_test_scenarios(self) -> Dict[str, Any]:
        """Load predefined test scenarios for benchmarking"""

        return {
            'code_review': {
                'name': 'Code Review',
                'description': 'Review code for quality, security, and best practices',
                'tasks': [
                    {
                        'id': 'auth_middleware',
                        'title': 'Authentication Middleware Review',
                        'code': '''
function authenticateUser(req, res, next) {
    const token = req.headers.authorization?.split(' ')[1];
    if (!token) {
        return res.status(401).json({ error: 'No token provided' });
    }

    try {
        const decoded = jwt.verify(token, process.env.JWT_SECRET);
        req.user = decoded;
        next();
    } catch (error) {
        return res.status(401).json({ error: 'Invalid token' });
    }
}

module.exports = authenticateUser;
                        ''',
                        'language': 'javascript',
                        'context': 'Node.js/Express authentication middleware',
                        'expected_issues': ['error_handling', 'security', 'validation']
                    },
                    {
                        'id': 'react_component',
                        'title': 'React Component Performance',
                        'code': '''
function UserList({ users, onUserSelect }) {
    const [selectedUser, setSelectedUser] = useState(null);

    const handleUserClick = (user) => {
        setSelectedUser(user);
        onUserSelect(user);
    };

    return (
        <div className="user-list">
            {users.map(user => (
                <div
                    key={user.id}
                    className={`user-item ${selectedUser?.id === user.id ? 'selected' : ''}`}
                    onClick={() => handleUserClick(user)}
                >
                    <img src={user.avatar} alt={user.name} />
                    <span>{user.name}</span>
                </div>
            ))}
        </div>
    );
}
                        ''',
                        'language': 'typescript',
                        'context': 'React component with performance considerations',
                        'expected_issues': ['performance', 'accessibility', 'state_management']
                    }
                ]
            },

            'debugging': {
                'name': 'Debugging Assistance',
                'description': 'Help debug complex issues with systematic approach',
                'tasks': [
                    {
                        'id': 'memory_leak',
                        'title': 'React Memory Leak',
                        'description': '''
Problem: React component is causing memory leaks. The component re-renders excessively and event listeners aren't being cleaned up properly.

Symptoms:
- Component re-renders on every state change
- Memory usage increases over time
- Browser becomes slow after extended use
- Console shows stale closure warnings

Code context:
- Uses useEffect for data fetching
- Has event listeners for window resize
- Manages local component state
- Receives props from parent component
                        ''',
                        'expected_solution_elements': ['useCallback', 'useMemo', 'cleanup', 'dependency_arrays']
                    },
                    {
                        'id': 'api_timeout',
                        'title': 'API Timeout Handling',
                        'description': '''
Problem: API calls are timing out inconsistently. Sometimes requests succeed, sometimes they fail with timeout errors.

Current implementation:
- Using fetch API with default timeout
- No retry logic
- Error handling is basic
- Network conditions vary (slow 3G, fast wifi)

Requirements:
- Handle network timeouts gracefully
- Implement retry logic with exponential backoff
- Provide user feedback during long requests
- Handle different types of network errors
                        ''',
                        'expected_solution_elements': ['timeout', 'retry', 'backoff', 'error_handling']
                    }
                ]
            },

            'architecture': {
                'name': 'System Architecture',
                'description': 'Design scalable and maintainable system architecture',
                'tasks': [
                    {
                        'id': 'microservices_design',
                        'title': 'E-commerce Microservices Architecture',
                        'requirements': '''
Design a microservices architecture for an e-commerce platform that must handle:
- 10,000+ concurrent users
- Product catalog with 1M+ items
- Real-time inventory updates
- Payment processing
- Order management
- User authentication and profiles
- Recommendation engine
- Analytics and reporting

Constraints:
- Must be cloud-native (AWS/GCP/Azure)
- Support for multiple regions
- 99.9% uptime requirement
- Budget-conscious scaling
- Team of 15 developers

Current tech stack: Node.js, React, PostgreSQL, Redis
                        ''',
                        'evaluation_criteria': ['scalability', 'reliability', 'maintainability', 'cost_effectiveness']
                    }
                ]
            },

            'testing': {
                'name': 'Testing Strategy',
                'description': 'Design comprehensive testing strategies',
                'tasks': [
                    {
                        'id': 'api_testing',
                        'title': 'REST API Testing Strategy',
                        'context': '''
API Endpoints to test:
- POST /api/users (user registration)
- GET /api/users/:id (get user profile)
- PUT /api/users/:id (update profile)
- DELETE /api/users/:id (delete account)
- POST /api/auth/login (authentication)
- POST /api/orders (create order)

Requirements:
- Test authentication and authorization
- Validate input data and error responses
- Test rate limiting and security
- Cover edge cases and error scenarios
- Integration with external payment service

Current setup: Jest, Supertest, PostgreSQL test database
                        ''',
                        'expected_coverage': ['unit', 'integration', 'e2e', 'security', 'performance']
                    }
                ]
            }
        }

    def load_collaboration_prompts(self) -> Dict[str, str]:
        """Load the specialized collaboration prompts"""

        prompt_file = "./prompts/human_ai_collaboration_system_20251102_083235.json"
        if os.path.exists(prompt_file):
            with open(prompt_file, 'r') as f:
                system = json.load(f)
            return system.get('prompts', {})
        return {}

    def evaluate_response_quality(self, response: str, task_type: str, expected_elements: List[str]) -> Dict[str, Any]:
        """Evaluate the quality of AI collaboration response"""

        metrics = {
            'length': len(response),
            'has_structure': self.check_structured_response(response),
            'relevance_score': self.calculate_relevance(response, task_type),
            'completeness_score': self.check_completeness(response, expected_elements),
            'actionability_score': self.check_actionability(response),
            'clarity_score': self.check_clarity(response),
            'expertise_score': self.check_expertise(response, task_type)
        }

        # Overall quality score (weighted average)
        weights = {
            'relevance_score': 0.25,
            'completeness_score': 0.25,
            'actionability_score': 0.2,
            'clarity_score': 0.15,
            'expertise_score': 0.15
        }

        overall_score = sum(metrics[key] * weights[key] for key in weights.keys())
        metrics['overall_quality'] = overall_score

        return metrics

    def check_structured_response(self, response: str) -> bool:
        """Check if response has clear structure and organization"""
        indicators = ['step', 'first', 'consider', 'recommend', 'summary', 'conclusion', '•', '1.', '2.', '3.']
        return any(indicator.lower() in response.lower() for indicator in indicators)

    def calculate_relevance(self, response: str, task_type: str) -> float:
        """Calculate how relevant the response is to the task type"""
        relevance_keywords = {
            'code_review': ['security', 'performance', 'maintainability', 'best practices', 'improvement'],
            'debugging': ['cause', 'solution', 'fix', 'debug', 'investigate', 'check'],
            'architecture': ['scalability', 'microservices', 'design', 'pattern', 'infrastructure'],
            'testing': ['test', 'coverage', 'strategy', 'validation', 'assert', 'mock']
        }

        keywords = relevance_keywords.get(task_type, [])
        matches = sum(1 for keyword in keywords if keyword.lower() in response.lower())
        return min(matches / len(keywords), 1.0) if keywords else 0.5

    def check_completeness(self, response: str, expected_elements: List[str]) -> float:
        """Check if response covers expected elements"""
        if not expected_elements:
            return 0.8  # Default good score if no specific expectations

        matches = sum(1 for element in expected_elements if element.lower() in response.lower())
        return matches / len(expected_elements)

    def check_actionability(self, response: str) -> float:
        """Check if response provides actionable guidance"""
        action_indicators = ['implement', 'change', 'update', 'add', 'remove', 'use', 'create', 'modify']
        matches = sum(1 for indicator in action_indicators if indicator in response.lower())
        return min(matches / 3, 1.0)  # At least 3 action items for full score

    def check_clarity(self, response: str) -> float:
        """Check clarity and readability of response"""
        # Penalize for excessive jargon, reward for clear explanations
        clarity_indicators = ['clearly', 'simply', 'example', 'instance', 'specifically']
        jargon_penalties = ['buzzword', 'complicated', 'complex']  # Simplified check

        clarity_score = sum(1 for indicator in clarity_indicators if indicator in response.lower()) / len(clarity_indicators)
        jargon_penalty = sum(1 for penalty in jargon_penalties if penalty in response.lower()) * 0.1

        return max(0, clarity_score - jargon_penalty)

    def check_expertise(self, response: str, task_type: str) -> float:
        """Check if response demonstrates domain expertise"""
        expertise_indicators = {
            'code_review': ['security', 'performance', 'maintainability', 'patterns'],
            'debugging': ['root cause', 'systematic', 'isolate', 'reproduce'],
            'architecture': ['scalability', 'reliability', 'trade-offs', 'patterns'],
            'testing': ['coverage', 'edge cases', 'integration', 'validation']
        }

        indicators = expertise_indicators.get(task_type, [])
        matches = sum(1 for indicator in indicators if indicator.lower() in response.lower())
        return min(matches / len(indicators), 1.0) if indicators else 0.7

    async def run_single_test(self, model_name: str, task_type: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single benchmark test"""

        model_config = self.models[model_name]
        prompts = self.load_collaboration_prompts()

        # Select appropriate prompt
        prompt_key = f"collaboration_{task_type}"
        system_prompt = prompts.get(prompt_key, prompts.get('base_collaboration', ''))

        # Prepare user message based on task type
        if task_type == 'code_review':
            user_message = f"Please review this {task.get('language', 'code')} code:\n\n```{task['language']}\n{task['code']}\n```"
            if task.get('context'):
                user_message += f"\n\nContext: {task['context']}"
        elif task_type == 'debugging':
            user_message = task['description']
        elif task_type == 'architecture':
            user_message = task['requirements']
        elif task_type == 'testing':
            user_message = f"Design a testing strategy for: {task.get('context', task.get('title', 'this system'))}"
        else:
            user_message = task.get('description', 'Please help with this task')

        start_time = time.time()

        try:
            if model_config['provider'] == 'ollama':
                response = ollama.chat(
                    model=model_config['model'],
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_message}
                    ],
                    options={'temperature': 0.7, 'num_predict': 1000}
                )
                content = response['message']['content']
            else:
                # OpenAI implementation would go here
                content = "OpenAI integration not implemented yet"

            latency = time.time() - start_time

            # Evaluate response quality
            expected_elements = task.get('expected_issues', []) + task.get('expected_solution_elements', []) + task.get('evaluation_criteria', []) + task.get('expected_coverage', [])
            quality_metrics = self.evaluate_response_quality(content, task_type, expected_elements)

            return {
                'success': True,
                'model': model_name,
                'task_type': task_type,
                'task_id': task['id'],
                'response': content,
                'latency': latency,
                'quality_metrics': quality_metrics,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            return {
                'success': False,
                'model': model_name,
                'task_type': task_type,
                'task_id': task['id'],
                'error': str(e),
                'latency': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }

    async def run_benchmark_suite(self, models_to_test: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run complete benchmark suite across all models and scenarios"""

        models_to_test = models_to_test or list(self.models.keys())
        results = {}

        print("🚀 Starting Human-AI Collaboration Benchmark Suite")
        print(f"📊 Testing {len(models_to_test)} models across {len(self.test_scenarios)} scenarios")
        print("=" * 80)

        for model_name in models_to_test:
            if model_name not in self.models:
                print(f"⚠️  Model {model_name} not available, skipping")
                continue

            print(f"\n🤖 Testing model: {model_name}")
            model_results = {}

            for scenario_name, scenario in self.test_scenarios.items():
                print(f"  📝 Running {scenario['name']} scenario...")

                scenario_results = []
                for task in scenario['tasks']:
                    print(f"    🧪 Task: {task['title']}")

                    result = await self.run_single_test(model_name, scenario_name, task)
                    scenario_results.append(result)

                    status = "✅" if result['success'] else "❌"
                    quality = result.get('quality_metrics', {}).get('overall_quality', 0)
                    print(f"      {status} Quality: {quality:.2f}")
                model_results[scenario_name] = scenario_results

            results[model_name] = model_results

        # Save results
        self.save_results(results)

        # Generate summary report
        summary = self.generate_summary_report(results)
        self.save_summary(summary)

        return results

    def save_results(self, results: Dict[str, Any]):
        """Save detailed benchmark results"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./benchmarks/benchmark_results_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"💾 Detailed results saved to: {filename}")

    def generate_summary_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary report"""

        summary = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'models_tested': list(results.keys()),
                'scenarios_tested': list(self.test_scenarios.keys()),
                'total_tests': 0
            },
            'model_performance': {},
            'scenario_performance': {},
            'recommendations': []
        }

        # Calculate per-model performance
        for model_name, model_results in results.items():
            model_metrics = {
                'total_tests': 0,
                'successful_tests': 0,
                'avg_quality_score': 0,
                'avg_latency': 0,
                'quality_scores': [],
                'latencies': []
            }

            for scenario_name, scenario_results in model_results.items():
                for result in scenario_results:
                    model_metrics['total_tests'] += 1
                    summary['metadata']['total_tests'] += 1

                    if result['success']:
                        model_metrics['successful_tests'] += 1
                        quality = result.get('quality_metrics', {}).get('overall_quality', 0)
                        latency = result.get('latency', 0)

                        model_metrics['quality_scores'].append(quality)
                        model_metrics['latencies'].append(latency)

            if model_metrics['quality_scores']:
                model_metrics['avg_quality_score'] = statistics.mean(model_metrics['quality_scores'])
                model_metrics['avg_latency'] = statistics.mean(model_metrics['latencies'])

            summary['model_performance'][model_name] = model_metrics

        # Calculate per-scenario performance
        for scenario_name in self.test_scenarios.keys():
            scenario_metrics = {
                'models_tested': 0,
                'avg_quality_by_model': {},
                'best_performing_model': None,
                'best_quality_score': 0
            }

            for model_name, model_results in results.items():
                if scenario_name in model_results:
                    scenario_results = model_results[scenario_name]
                    qualities = [
                        r.get('quality_metrics', {}).get('overall_quality', 0)
                        for r in scenario_results if r['success']
                    ]

                    if qualities:
                        avg_quality = statistics.mean(qualities)
                        scenario_metrics['avg_quality_by_model'][model_name] = avg_quality
                        scenario_metrics['models_tested'] += 1

                        if avg_quality > scenario_metrics['best_quality_score']:
                            scenario_metrics['best_quality_score'] = avg_quality
                            scenario_metrics['best_performing_model'] = model_name

            summary['scenario_performance'][scenario_name] = scenario_metrics

        # Generate recommendations
        summary['recommendations'] = self.generate_recommendations(summary)

        return summary

    def generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on results"""

        recommendations = []

        # Find best overall model
        best_model = None
        best_score = 0

        for model_name, metrics in summary['model_performance'].items():
            if metrics['avg_quality_score'] > best_score:
                best_score = metrics['avg_quality_score']
                best_model = model_name

        if best_model:
            recommendations.append(f"🏆 Use {best_model} as the primary collaboration model (avg quality: {best_score:.2f})")

        # Identify scenarios needing improvement
        for scenario_name, metrics in summary['scenario_performance'].items():
            if metrics['best_quality_score'] < 0.7:
                recommendations.append(f"🔧 Improve {scenario_name} scenario - current best score: {metrics['best_quality_score']:.2f}")

        # Latency considerations
        for model_name, metrics in summary['model_performance'].items():
            if metrics['avg_latency'] > 10:  # More than 10 seconds
                recommendations.append(f"⚡ Optimize {model_name} latency (avg: {metrics['avg_latency']:.1f}s)")

        # Success rate
        for model_name, metrics in summary['model_performance'].items():
            success_rate = metrics['successful_tests'] / metrics['total_tests'] if metrics['total_tests'] > 0 else 0
            if success_rate < 0.9:
                recommendations.append(f"🔧 Improve {model_name} reliability ({success_rate:.1%} success rate)")

        return recommendations

    def save_summary(self, summary: Dict[str, Any]):
        """Save summary report"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./benchmarks/benchmark_summary_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"📊 Summary report saved to: {filename}")

        # Print key findings
        print("\n🎯 Key Findings:")
        print("-" * 40)

        for model_name, metrics in summary['model_performance'].items():
            print(f"  {model_name}: {metrics['avg_quality_score']:.2f} quality, {metrics['avg_latency']:.1f}s latency")
        print("\n💡 Recommendations:")
        for rec in summary['recommendations']:
            print(f"  {rec}")

async def main():
    """Run the benchmark suite"""

    benchmark = CollaborationBenchmark()

    # Run benchmarks for available models
    available_models = []
    for model_name, config in benchmark.models.items():
        if config['provider'] == 'ollama':
            try:
                ollama.show(model_name)
                available_models.append(model_name)
                print(f"✅ Model {model_name} is available")
            except:
                print(f"⚠️  Model {model_name} not available")
        else:
            available_models.append(model_name)

    if not available_models:
        print("❌ No models available for testing")
        return 1

    results = await benchmark.run_benchmark_suite(available_models)

    print("\n🎉 Benchmarking complete!")
    print("📈 Results saved to ./benchmarks/ directory")

    return 0

if __name__ == "__main__":
    import asyncio
    exit_code = asyncio.run(main())
    exit(exit_code)