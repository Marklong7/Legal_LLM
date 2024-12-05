# Legal_LLM

## Project Summary:
This project focuses on enhancing LLM performance on Legal questions by integrating advanced techniques such as few-shot prompt engineering, Chain-of-Thought (CoT) prompting, ReAct, adaptive RAG, and RAPTOR. Our system achieves a significant 110% improvement over baseline models on Illinois Law question-answering.

## Core Contributions:
* Benchmark knowledge-Informed Adaptive RAG:
Inspired by the Adaptive RAG paper, which fine-tuned an LLM classifier to dynamically adjust the retrieval strategy, we take this concept further. Our approach incorporates knowledge from public legal benchmarks (e.g., LegalBench) to enhance the accuracy of the LLM classifier via CoT prompt engineering. This allows the system to more effectively determine when to utilize RAG or other tools based on the predicted difficulty of the question.

* ReAct and advanced RAG for Complex Questions:
For questions classified as "hard", our system employs ReAct to promote deeper reasoning.  This involves strategically combining ReAct with techniques like RAPTOR, Google Search, and self-reflection to gather and synthesize relevant information. RAPTOR is essentially a data augmentation technique, it enhances RAG performance by clustering-summarizing to generate additional documents with high-level information. We rewrite the code of RAPTOR, optimized for GPU acceleration and parallelism to improve efficiency.
