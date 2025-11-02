#!/usr/bin/env python3
"""
Simple Real Test - One Scenario Only
"""

import json
import time
import ollama
from datetime import datetime

def load_prompts():
    """Load our specialized prompts"""
    try:
        with open("./prompts/human_ai_collaboration_system_20251102_083235.json", 'r') as f:
            system = json.load(f)
        return system.get('prompts', {})
    except:
        return {}

def test_code_review():
    """Test just one code review scenario"""

    print("🧪 Testing REAL Code Review with granite4:3b")
    print("=" * 50)

    # Load prompts
    prompts = load_prompts()
    system_prompt = prompts.get('code_review_specialist', 'You are a helpful code reviewer.')

    # Test code
    code = '''
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
'''

    user_prompt = f'''Please review this authentication middleware code for security issues, performance problems, and best practices:

```javascript
{code}
```

What issues do you see and how would you improve this code?'''

    print("📝 Sending to granite4:3b...")
    start_time = time.time()

    try:
        response = ollama.chat(
            model="granite4:3b",
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            options={'temperature': 0.7, 'num_predict': 600}
        )

        latency = time.time() - start_time
        content = response['message']['content']

        print("✅ Response received!"        print(".1f"        print(f"Response length: {len(content)} characters")
        print("\n📄 AI Response:")
        print("-" * 30)
        print(content)
        print("-" * 30)

        # Save result
        result = {
            'test': 'code_review_auth_middleware',
            'model': 'granite4:3b',
            'success': True,
            'latency': round(latency, 2),
            'response_length': len(content),
            'response': content,
            'timestamp': datetime.now().isoformat(),
            'prompt_used': 'code_review_specialist'
        }

        with open('./benchmarks/real_test_result.json', 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print("💾 Result saved to: ./benchmarks/real_test_result.json")

        return result

    except Exception as e:
        latency = time.time() - start_time
        print(f"❌ Test failed: {e}")
        print(".1f"
        return {'success': False, 'error': str(e), 'latency': round(latency, 2)}

if __name__ == "__main__":
    result = test_code_review()

    if result['success']:
        print("\n🎉 REAL TEST COMPLETE!")
        print("This is an actual result from granite4:3b with our specialized prompts.")
        print("No fabrication - just honest benchmarking.")
    else:
        print("\n❌ Test failed - but at least we tried with real data!")