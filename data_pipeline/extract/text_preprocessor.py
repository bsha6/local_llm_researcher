import re
import unicodedata

class TextPreprocessor:
    def __init__(self, text, mode="passage"):
        """
        Initializes the preprocessor.
        :param text: The input text to be cleaned.
        :param mode: "query" for search queries, "passage" for documents.
        """
        assert mode in ["query", "passage"], "Mode must be 'query' or 'passage'."
        self.text = text
        self.mode = mode

    def clean_text(self):
        """Performs text normalization: removes unnecessary spaces, normalizes Unicode, and lowercases text."""
        text = self.text.strip().lower()  # Lowercase & trim
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        
        # Remove footnotes like [1], (1), or **1.
        text = re.sub(r"\[\d+\]|\(\d+\)|\*\d+", "", text)

        # Remove URLs but keep DOI links & references
        text = re.sub(r"https?://\S+", "", text)  # Remove web links

        # Remove page numbers (e.g., "Page 23" or "- 45 -")
        text = re.sub(r"(page\s*\d+|\-\s*\d+\s*\-)", "", text, flags=re.IGNORECASE)

        # Remove common headers (modify if necessary for your dataset)
        text = re.sub(r"(introduction|conclusion|references)", "", text, flags=re.IGNORECASE)

        # Remove "......"
        text = re.sub(r'(\s*\.\s*){3,}', ' ', text)

        # Remove numbered sections
        text = re.sub(r'^\d+(\.\d+)*\s+', '', text, flags=re.MULTILINE)

        # Collapse multiple spaces & trim text
        text = re.sub(r'\s+', ' ', text).strip()

        return text


# Example usage:
if __name__ == "__main__":
    # Simple test
    sample_text = """"Contents\n1\nIntroduction\n3\n1.1\nContributions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .\n4\n1.2\nSummary of Evaluation Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . .\n4\n2\nApproach\n5\n2.1\nOverview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .\n5\n2.2\nDeepSeek-R1-Zero: Reinforcement Learning on the Base Model . . . . . . . . . .\n5\n2.2.1\nReinforcement Learning Algorithm\n. . . . . . . . . . . . . . . . . . . . . .\n5\n2.2.2\nReward Modeling\n. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .\n6\n2.2.3\nTraining Template\n. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .\n6\n2.2.4\nPerformance, Self-evolution Process and Aha Moment of DeepSeek-R1-Zero\n6\n2.3\nDeepSeek-R1: Reinforcement Learning with Cold Start . . . . . . . . . . . . . . .\n9\n2.3.1\nCold Start . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .\n9\n2.3.2\nReasoning-oriented Reinforcement Learning . . . . . . . . . . . . . . . . .\n10\n2.3.3\nRejection Sampling and Supervised Fine-Tuning . . . . . . . . . . . . . . .\n10\n2.3.4\nReinforcement Learning for all Scenarios . . . . . . . . . . . . . . . . . . .\n11\n2.4\nDistillation: Empower Small Models with Reasoning Capability . . . . . . . . . .\n11\n3\nExperiment\n11\n3.1\nDeepSeek-R1 Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .\n13\n3.2\nDistilled Model Evaluation\n. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .\n14\n4\nDiscussion\n14\n4.1\nDistillation v.s. Reinforcement Learning . . . . . . . . . . . . . . . . . . . . . . . .\n14\n4.2\nUnsuccessful Attempts . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .\n15\n5\nConclusion, Limitations, and Future Work\n16\nA Contributions and Acknowledgments\n20\n2",
        "2": "1. Introduction\nIn recent years, Large Language Models (LLMs) have been undergoing rapid iteration and\nevolution (Anthropic, 2024; Google, 2024; OpenAI, 2024a), progressively diminishing the gap\ntowards Arti\ufb01cial General Intelligence (AGI).\nRecently, post-training has emerged as an important component of the full training pipeline.\nIt has been shown to enhance accuracy on reasoning tasks, align with social values, and adapt\nto user preferences, all while requiring relatively minimal computational resources against\npre-training. In the context of reasoning capabilities, OpenAI\u2019s o1 (OpenAI, 2024b) series models\nwere the \ufb01rst to introduce inference-time scaling by increasing the length of the Chain-of-\nThought reasoning process. This approach has achieved signi\ufb01cant improvements in various\nreasoning tasks, such as mathematics, coding, and scienti\ufb01c reasoning. However, the challenge\nof effective test-time scaling remains an open question for the research community. Several prior\nworks have explored various approaches, including process-based reward models (Lightman\net al., 2023; Uesato et al., 2022; Wang et al., 2023), reinforcement learning (Kumar et al., 2024),\nand search algorithms such as Monte Carlo Tree Search and Beam Search (Feng et al., 2024; Trinh\net al., 2024; Xin et al., 2024). However, none of these methods has achieved general reasoning\nperformance comparable to OpenAI\u2019s o1 series models.\nIn this paper, we take the \ufb01rst step toward improving language model reasoning capabilities\nusing pure reinforcement learning (RL). Our goal is to explore the potential of LLMs to develop\nreasoning capabilities without any supervised data, focusing on their self-evolution through\na pure RL process. Speci\ufb01cally, we use DeepSeek-V3-Base as the base model and employ\nGRPO (Shao et al., 2024) as the RL framework to improve model performance in reasoning.\nDuring training, DeepSeek-R1-Zero naturally emerged with numerous powerful and interesting\nreasoning behaviors. After thousands of RL steps, DeepSeek-R1-Zero exhibits super performance\non reasoning benchmarks. For instance, the pass@1 score on AIME 2024 increases from 15.6% to\n71.0%, and with majority voting, the score further improves to 86.7%, matching the performance\nof OpenAI-o1-0912.\nHowever, DeepSeek-R1-Zero encounters challenges such as poor readability, and language\nmixing. To address these issues and further enhance reasoning performance, we introduce\nDeepSeek-R1, which incorporates a small amount of cold-start data and a multi-stage training\npipeline. Speci\ufb01cally, we begin by collecting thousands of cold-start data to \ufb01ne-tune the\nDeepSeek-V3-Base model. Following this, we perform reasoning-oriented RL like DeepSeek-R1-\nZero. Upon nearing convergence in the RL process, we create new SFT data through rejection\nsampling on the RL checkpoint, combined with supervised data from DeepSeek-V3 in domains\nsuch as writing, factual QA, and self-cognition, and then retrain the DeepSeek-V3-Base model.\nAfter \ufb01ne-tuning with the new data, the checkpoint undergoes an additional RL process, taking\ninto account prompts from all scenarios. After these steps, we obtained a checkpoint referred to\nas DeepSeek-R1, which achieves performance on par with OpenAI-o1-1217.\nWe further explore distillation from DeepSeek-R1 to smaller dense models. Using Qwen2.5-\n32B (Qwen, 2024b) as the base model, direct distillation from DeepSeek-R1 outperforms applying\nRL on it. This demonstrates that the reasoning patterns discovered by larger base models are cru-\ncial for improving reasoning capabilities. We open-source the distilled Qwen and Llama (Dubey\net al., 2024) series. Notably, our distilled 14B model outperforms state-of-the-art open-source\nQwQ-32B-Preview (Qwen, 2024a) by a large margin, and the distilled 32B and 70B models set a\nnew record on the reasoning benchmarks among dense models.\n3","""
    # "   这是 一个 测试    文本。This is a test sentence!   "
    
    processor = TextPreprocessor(sample_text, mode="passage")
    cleaned_text = processor.clean_text()
    
    print("Processed Text:", cleaned_text)