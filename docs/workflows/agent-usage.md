# Agent Usage Guide

Complete guide to using the specialized subagents in LLM-Local-Lab.

**Last Updated**: 2025-10-02

---

## Overview

This project includes 4 specialized AI subagents that automate common tasks in LLM experimentation. Subagents are invoked within Claude Code conversations using the `@agent-name` syntax.

**Available Agents**:
1. `@benchmark-analyst` - Analyze benchmark results
2. `@model-configurator` - Create optimal model configurations
3. `@documentation-writer` - Generate and update documentation
4. `@gpu-optimizer` - Diagnose and optimize GPU performance

---

## How Subagents Work

### Invocation Syntax

```
@agent-name [your request]
```

**Example**:
```
@benchmark-analyst analyze the latest Llama 70B results
```

### When to Use Subagents

**Use subagents when**:
- Task requires specialized expertise
- You want focused analysis without general context
- Automating repetitive documentation/analysis tasks

**Don't use subagents when**:
- Simple one-liner questions
- Context from main conversation is critical
- Quick file reads or searches

### Subagent Capabilities

Each subagent has:
- **Specialized system prompt**: Expert knowledge in their domain
- **Limited tool access**: Only tools needed for their task
- **Separate context**: Doesn't carry over main conversation history
- **Focused output**: Produces actionable results

---

## @benchmark-analyst

**Purpose**: Analyze LLM benchmark results and generate insights

### When to Use

- After running benchmarks to understand results
- Comparing multiple benchmark runs
- Detecting performance patterns or anomalies
- Generating performance reports

### Example Invocations

```
# Analyze latest results
@benchmark-analyst analyze the most recent benchmark in results/raw/

# Compare runs
@benchmark-analyst compare the last 3 Llama benchmarks and identify trends

# Investigate issue
@benchmark-analyst why is GPU 1 showing lower utilization in the latest run?

# Weekly summary
@benchmark-analyst summarize all benchmarks from the past week
```

### What It Produces

- **Performance summaries**: Mean/median/std metrics
- **Comparative analysis**: Run-to-run comparisons
- **Insights**: What the numbers mean, good/bad/expected
- **Recommendations**: Next steps based on findings

### Example Output Structure

```markdown
# Benchmark Analysis: Llama-3.1-70B-Q4

## Executive Summary
- Throughput: 45 tokens/s (within expected range for Q4)
- VRAM: 42GB across dual-GPU (well-utilized)
- Recommendation: Try Q5 for quality improvement

## Detailed Metrics
[Tables and analysis]

## Observations
[Key findings]

## Recommendations
[Actionable next steps]
```

---

## @model-configurator

**Purpose**: Create optimal configurations for models on your hardware

### When to Use

- Adding a new model to test
- Troubleshooting OOM (out of memory) errors
- Optimizing existing configurations
- Deciding between single/dual GPU

### Example Invocations

```
# Configure new model
@model-configurator create a config for Mixtral 8x7B on dual RTX 5090

# Fix OOM
@model-configurator I'm getting OOM with Qwen 72B Q5, suggest fixes

# Optimize existing
@model-configurator optimize this config for maximum throughput
[paste YAML config]

# Compare strategies
@model-configurator should I use single or dual GPU for Llama 34B Q4?
```

### What It Produces

- **YAML configs**: Ready-to-use model configurations
- **VRAM estimates**: Predicted memory usage
- **Optimization suggestions**: Batch size, quantization, parallelism
- **Troubleshooting steps**: How to fix OOM or performance issues

### Example Output

```yaml
model:
  name: "mixtral-8x7b"
  variant: "q5-dual-gpu"
  model_id: "mistralai/Mixtral-8x7B-v0.1"

quantization:
  type: "q5_k_m"
  bits: 5

hardware:
  gpu_allocation:
    - device: 0
      layers: "0-15"
    - device: 1
      layers: "16-31"

inference:
  batch_size: 8
  max_context_length: 8192
  flash_attention: true
```

---

## @documentation-writer

**Purpose**: Generate and maintain project documentation

### When to Use

- After completing benchmarks or experiments
- Updating the models registry
- Creating weekly/monthly reports
- Documenting new findings or configurations

### Example Invocations

```
# Document benchmark
@documentation-writer create a report for the latest Llama 70B benchmark

# Update registry
@documentation-writer add Mixtral 8x7B to the models registry with these results:
[paste key metrics]

# Weekly report
@documentation-writer generate a weekly report for all experiments this week

# Update docs
@documentation-writer update the README with the new baseline performance numbers
```

### What It Produces

- **Benchmark reports**: Formatted markdown in `benchmarks/results/reports/`
- **Registry entries**: Updates to `docs/models/models-registry.md`
- **Guides and tutorials**: New documentation files
- **Changelogs**: Summary of changes and findings

### Example Output

```markdown
## Mixtral 8x7B - Q5

**Tested**: 2025-10-02
**Hardware**: Dual RTX 5090
**Quantization**: Q5_K_M
**VRAM Usage**: 32GB (GPU0: 16GB, GPU1: 16GB)

**Performance**:
- Throughput: 68 tokens/s
- TTFT: 85ms
- Batch size: 8

**Recommendation**: ⭐⭐⭐⭐ (4/5) - Excellent for coding tasks
```

---

## @gpu-optimizer

**Purpose**: Diagnose and optimize GPU performance

### When to Use

- Before running large benchmarks (pre-flight check)
- Investigating performance issues
- GPU underutilization or thermal problems
- Configuring multi-GPU strategies

### Example Invocations

```
# Pre-flight check
@gpu-optimizer verify system is ready for benchmark

# Diagnose issue
@gpu-optimizer why is throughput so low for this 7B model?

# Balance GPUs
@gpu-optimizer GPU 0 is at 100% but GPU 1 is at 30%, how to fix?

# Thermal check
@gpu-optimizer check if GPUs are throttling during this benchmark
```

### What It Produces

- **System status reports**: GPU utilization, temps, VRAM
- **Diagnostic insights**: Root cause of performance issues
- **Optimization recommendations**: Concrete steps to improve
- **Pre-flight validation**: Go/no-go for benchmark execution

### Example Output

```
=== GPU Pre-Flight Check ===

GPU 0: RTX 5090
  Utilization: 2% (IDLE - OK)
  Temperature: 38C (COOL - OK)
  VRAM: 31.8GB free (READY)

GPU 1: RTX 5090
  Utilization: 1% (IDLE - OK)
  Temperature: 36C (COOL - OK)
  VRAM: 31.9GB free (READY)

Status: READY FOR BENCHMARK

Recommendations:
- Clear CUDA cache before benchmark
- Monitor temps during run (target <80C)
```

---

## Workflow Examples

### Workflow 1: Testing a New Model

```
User: I want to test Llama 3.1 8B

@model-configurator create optimal config for Llama 3.1 8B on RTX 5090

[Agent generates config YAML]

@gpu-optimizer check if system is ready

[Agent validates GPUs are idle]

User: Run the benchmark
[Execute: python benchmarks/scripts/llm_inference_benchmark.py --model meta-llama/Llama-3.1-8B ...]

@benchmark-analyst analyze the results in results/raw/[latest].json

[Agent generates performance report]

@documentation-writer add Llama 3.1 8B to the models registry with these results

[Agent updates registry]
```

### Workflow 2: Troubleshooting OOM

```
User: Getting OOM with Qwen 72B Q5

@model-configurator diagnose why Qwen 72B Q5 is causing OOM

[Agent calculates VRAM needs, suggests Q4 or dual-GPU]

User: Try Q4 instead

@model-configurator create Q4 config for Qwen 72B dual-GPU

[Agent generates new config]

@gpu-optimizer validate this config will fit in VRAM

[Agent confirms with calculations]
```

### Workflow 3: Weekly Reporting

```
User: Generate weekly report

@benchmark-analyst summarize all benchmarks from Oct 1-7

[Agent analyzes all runs, finds patterns]

@documentation-writer create a weekly report with these insights:
[paste analyst output]

[Agent generates formatted report in docs/weekly-reports/]
```

---

## Tips and Best Practices

### Be Specific

**Good**: `@benchmark-analyst compare throughput between Q4 and Q5 Llama 70B`
**Bad**: `@benchmark-analyst analyze stuff`

### Provide Context

**Good**: `@model-configurator I'm getting OOM with this config: [paste YAML]`
**Bad**: `@model-configurator fix my OOM`

### Use the Right Agent

- **Analyzing numbers** → `@benchmark-analyst`
- **Creating configs** → `@model-configurator`
- **Writing docs** → `@documentation-writer`
- **GPU issues** → `@gpu-optimizer`

### Delegate Complex Tasks

Don't manually parse JSON or write reports. Let agents do it:

```
# Instead of reading JSON yourself
User: [reads benchmark file manually, tries to interpret]

# Delegate to specialist
@benchmark-analyst analyze benchmarks/results/raw/20251002_llama70b.json
```

### Chain Agents

Use output from one agent as input to another:

```
@benchmark-analyst analyze latest benchmark
[Get insights]

@documentation-writer create report with these insights: [paste]
```

---

## Advanced Usage

### Creating Custom Agents

You can create your own subagents in `.claude/agents/`:

1. Create `my-agent.md` in `.claude/agents/`
2. Add YAML frontmatter:
   ```yaml
   ---
   name: my-agent
   description: What this agent does
   tools: [Read, Write, Bash]
   model: sonnet
   ---
   ```
3. Write detailed system prompt describing expertise
4. Invoke with `@my-agent`

### Modifying Existing Agents

Edit the `.md` files in `.claude/agents/` to:
- Add new capabilities
- Change tool permissions
- Refine system prompts
- Switch model (sonnet, opus, haiku)

### Debugging Agent Issues

If an agent isn't working:

1. Check agent file exists in `.claude/agents/`
2. Validate YAML frontmatter syntax
3. Ensure tools are specified correctly
4. Try explicit invocation: `@agent-name do X with file Y`

---

## Agent Comparison Matrix

| Feature | benchmark-analyst | model-configurator | documentation-writer | gpu-optimizer |
|---------|------------------|-------------------|---------------------|---------------|
| **Primary Task** | Analyze results | Create configs | Write docs | Optimize GPUs |
| **Tools** | Read, Bash, Write, Grep | Read, Write, Edit, Bash | Read, Write, Edit, Glob | Bash, Read, Grep |
| **Output** | Reports, insights | YAML configs | Markdown docs | Diagnostics, fixes |
| **When to Use** | Post-benchmark | Pre-benchmark | Any time | Pre-benchmark, troubleshooting |
| **Expertise** | Statistics, performance | Hardware, optimization | Writing, formatting | CUDA, diagnostics |

---

## Frequently Asked Questions

**Q: Can I use multiple agents in one message?**
A: Yes! Invoke them separately or sequentially:
```
@gpu-optimizer check system
@benchmark-analyst analyze latest
```

**Q: Do agents share context?**
A: No, each agent invocation is independent. Provide all needed context.

**Q: Which model do agents use?**
A: By default `sonnet`. You can change in the agent's YAML frontmatter.

**Q: Can I disable an agent?**
A: Rename or delete the `.md` file in `.claude/agents/`

**Q: How do I see what tools an agent has?**
A: Check the `tools:` array in the YAML frontmatter of the agent file.

---

**Next Steps**:
- Try invoking `@gpu-optimizer check system` to validate setup
- Run a benchmark and use `@benchmark-analyst` to analyze it
- Experiment with creating your own custom agents
