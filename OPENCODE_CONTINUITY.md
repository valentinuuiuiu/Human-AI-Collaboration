# OpenCode Continuity Guide

## How to Maintain Context with OpenCode

### 📝 Session Continuity Methods

1. **Create Context Files**
   - `CONTEXT.md` - Current project status and goals
   - `TODO.md` - Active tasks and priorities
   - `PROGRESS.md` - What was accomplished last session

2. **Use Todo Lists**
   - OpenCode has a built-in `todowrite` tool for tracking tasks
   - Tasks persist across sessions and can be resumed

3. **Project Documentation**
   - Keep README files updated
   - Document architecture decisions
   - Track important file locations and patterns

### 🔄 Session Workflow

**Start each session by:**
1. Reading your context files
2. Checking the todo list
3. Reviewing recent progress

**End each session by:**
1. Updating context files
2. Adding new todos for next session
3. Documenting what was accomplished

### 📁 Recommended File Structure

```
project/
├── CONTEXT.md           # Current session context
├── TODO.md             # Active tasks
├── PROGRESS.md         # Session summaries
├── docs/               # Project documentation
└── sessions/           # Optional: session notes
    ├── session-2024-11-02.md
    └── session-2024-11-03.md
```

### 💡 Pro Tips

- **Don't rely on my memory** - I start fresh each session
- **Document important decisions** in files I can read
- **Use the todo system** for task tracking
- **Keep context files updated** after each session
- **Name files descriptively** so I can find them easily

### 🚀 Quick Start

Run this at the start of each session:
```bash
# Read current context
cat CONTEXT.md
cat TODO.md

# Check recent progress
cat PROGRESS.md
```

This will help me understand where we left off and what needs to be done next!