We are grateful for the constructive comments. To start with, please note that the repository is available at 

https://github.com/HPCL-Coder/HPCL

------------------------------

# Review A

### A-Q1

>How is the dataset built for languages/libraries/frameworks other than OpenMP?

This is an insightful question. For frameworks beyond OpenMP (e.g., CUDA, MPI), we construct serial-parallel code pairs by systematically removing the corresponding parallel statements from real-world parallel implementations. This simplification was necessary because it is extremely difficult to obtain real-world serial code that matches actual CUDA/MPI implementations in behavior and structure. As in prior works such as OMPar and MPIrigen, we adopt this practical workaround to enable large-scale dataset construction.

While this approach does not fully reflect the complexity of real development workflows, it enables controlled evaluation and reproducible analysis by isolating the parallelization aspect. Moreover, even under this simplified setup, current LLMs struggle to generate correct and effective parallel code—indicating that our benchmark still poses a meaningful challenge and establishes a lower bound for model capability. We will continue to enhance the dataset by manually constructing more realistic serial-parallel benchmarks that better reflect real-world development scenarios.


### A-Q2

>What is the input for CUDA/MPI cases?

For CUDA/MPI cases, the input is created by systematically removing CUDA/MPI statements from real parallel code. Specific input-output examples will be available on our GitHub repository [LINK](https://github.com/HPCL-Coder/HPCL/input_example.md).

### A-Q3

> What was the evaluation on cases that include more than one language/library/framework?  

Thank you for the insightful question. Our study primarily targets scenarios where multiple parallel frameworks coexist at the project level, which is common in real-world HPC software. However, functions that integrate multiple frameworks simultaneously (e.g., MPI+CUDA in a single function) are rare in practice, and only account for approximately 1% of our dataset.

In our evaluation, all frameworks are treated under a unified task formulation that includes Insertion Point (IP), Framework Selection (FW), and Directive Completion (DIR). This design ensures consistent task structure and fair comparison across different frameworks, regardless of whether they appear alone or in combination.

While multi-framework-in-one-function cases are few, our method has been shown to remain effective in these scenarios. We provide examples and results on our GitHub repository.  [LINK](https://github.com/HPCL-Coder/HPCL/blob/main/README.md#-single-framework-vs-multi-framework-evaluation).


 
------------------------------


# Review B

## B-Q1 

>How do you justify the use of the term “code completion” for a task that involves structured transformation and semantic parallelization? Wouldn't “transformation” or a similar term more accurately reflect the nature and objectives of the task?


Thank you for the insightful comment. While we used “code completion” to emphasize the automated insertion of missing parallel constructs, we agree that “transformation” better reflects the nature of the task. We will clarify the terminology to avoid ambiguity.



## B-Q2 

>Can you provide an evaluation of your approach’s practical performance in real-world deployment scenarios, using dynamic metrics such as execution speedup, memory usage, or compilation success rate?
 
Thank you for the valuable suggestion. We fully agree that dynamic metrics are crucial for assessing practical utility. While not yet included due to the challenges of automating cross-framework testing, dynamic evaluation is a key focus of our future work.


----------------------------------------------

# Review C


### C-Q1

>What decoding parameters were used for different LLMs? Were seeds fixed for reproducibility?

All LLMs used identical decoding parameters and a fixed random seed to ensure reproducibility. Full details are available on our GitHub repository [LINK](https://github.com/HPCL-Coder/HPCL/model_parameters.md).

### C-Q2

>What are the full training settings for SFT and HPCL models (e.g., hyperparameters, hardware)?


All training hyperparameters and hardware details for SFT and HPCL models are fully documented and will be available on our GitHub repository [LINK](https://github.com/HPCL-Coder/HPCL/model_parameters.md).

### C-Q3

>How does HPCL differ technically from standard curriculum learning methods?

HPCL extends standard curriculum learning in two key ways: (1) it integrates sample difficulty progression with phase-wise subtask learning, forming a hierarchical, progressive schedule; and (2) it supports multi-objective decomposition and stage-wise joint optimization, making it well-suited for structured, multi-stage code generation. This design enhances both local and global decision-making, enabling more effective handling of complex parallelization tasks.


### C-Q4

>Have you analyzed the differences in error distributions between HPCL and SFT?

This is an insightful question. We have indeed conducted a systematic analysis of error distributions for both HPCL and SFT. Although these results were omitted from the paper due to space constraints, our analysis shows that SFT produces a more dispersed error distribution, with notably higher proportions of insertion point errors, redundant or hallucinated code, missing parallel structures, and low-level detail mistakes. This indicates that SFT models are less robust and generalizable for parallelization tasks. The full error statistical details will be made available on our GitHub repository for your reference [LINK](https://github.com/HPCL-Coder/HPCL/SFT_error.md).

