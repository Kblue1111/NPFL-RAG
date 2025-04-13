# Overall

This repository contains the full implementation of **NPFL-RAG**, a novel framework that integrates LLMs with external knowledge retrieval for fault localization in novice programs. It includes all source code, utilities, and evaluation scripts necessary to replicate the experiments in our study.

![image](https://github.com/user-attachments/assets/76ec31bb-dd0c-4f3c-b5fc-ff0573250dff)


## 1. Requirements

Locate the **requirements.txt** file in the root directory and execute the following dependency command.

```
pip install -r requirements.txt
```

Should there be any additional dependencies missing, feel free to sequentially import them using pip based on the provided information.



## 2. LLMs

In this study, multiple different LLMs were used for experimentation, namely o1-preview, o3-mini, GPT-4o,
GPT-3.5, Claude3, Qwen2.5-turbo, Llama3,  ChatGLM3, DeepSeek-v3 and DeepSeek-r1.

Among these, Qwen2.5-turbo, Llama3,  ChatGLM3, DeepSeek-v3 and DeepSeek-r1 are open-source LLMs that can be deployed following their respective official documentation. The versions used and their corresponding official website links are provided below:

| Model Name                            | HggingFace Url                                               |
| ------------------------------------- | ------------------------------------------------------------ |
| **Qwen/Qwen2.5-Turbo-1M-Demo**        | **https://huggingface.co/spaces/Qwen/Qwen2.5-Turbo-1M-Demo** |
| **meta-llama/Llama-3.3-70B-Instruct** | **https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct** |
| **THUDM/chatglm3-6b**                 | **https://huggingface.co/THUDM/chatglm3-6b**                 |
| **deepseek-ai/DeepSeek-V3**           | **https://huggingface.co/deepseek-ai/DeepSeek-V3**           |
| **deepseek-ai/DeepSeek-R1**           | **https://huggingface.co/deepseek-ai/DeepSeek-R1**           |

For conducting experiments on commercial LLMs like o1-preview, o3-mini, GPT-4o, GPT-3.5 and Claude3, we utilize the official API interfaces provided.

To standardize our experiments, for all open-source LLMs, we employ the API interfaces in the format of commercial LLMs. Instructions on deploying a commercial LLM form API interface can be found at https://github.com/xusenlinzy/api-for-open-llm.



## 3. Dataset

We conduct our experiments on the TutorCode dataset. For detailed information and access, please refer to: <https://github.com/buaabarty/CREF>.



## 4. Run

To facilitate experiment reproducibility, we have organized the main code into the **RAGFL** directory. <u>Replicating the experiments only requires attention to this folder.</u>



- Navigate to the **RAGFL** directory:

  **Sendpromt** is used to send requests to LLMs and collect results.

  - Within it, the function *send_prompt_openai_gpt* is used for sending requests to GPT. In this function, the *base_url* and *api_key* need to be changed to correspond to the requested link address and key, which can be obtained by purchasing credits on the official website.
  - The *send_prompt_openai_form* function is used for requests to open-source LLMs. Simply modify the request address to the API address of the aforementioned LLM. The key can be filled in arbitrarily.
  - <u>It is advisable to test the Sendpromt requests for successful validation before proceeding to the next step.</u>

*ReadJsonTest* function is utilized to extract the JSON fields from the results returned by the LLM.

*getTokenNumbers* function is employed to extract the token count of the prompt.

*AddLineNumber* is responsible for processing the source code by adding line numbers to each line.

utils1 is employed to implement the retrieval component of RAG.

------

Gpt4o-send, Gpt3.5-send, GLM3-send, LLama3-send are employed to dispatch requests for novice program fault localization in bulk to various LLMs, subsequently storing the resulting data in the 'data' repository. To switch models, simply modify the value of `experiment_model`.

Before execution, certain configurations pertaining to your setup necessitate adjustments. For illustrative purposes, let us consider the instance of *Gpt4o-send*:

- The *'prompt_location'* parameter serves to acquire prompts and can be altered to accommodate a custom prompt of your choice.
- The *'run_TutorCode'* function involves traversing the respective datasets once for data retrieval, with the *<u>'root_path'</u>* requiring adjustment to reflect the location where the dataset is stored.
- *'TutorCode_Filter_Data.pkl'* houses the program information filtered from our dataset.



## 5. Evaluate

Navigate to the *"Evaluate"* directory for the code responsible for evaluating the results of the data.

### Accuracy Count

Enter the "Evaluate" directory to calculate the accuracy of each LLM in the 'total_count' file and to assess the accuracy of SBFL and MBFL in the 'sbfl_mbfl_count' file. <u>*Remember to adjust the 'root_path' to the appropriate dataset location before executing*</u>. If utilizing the dataset sources from the previous sections, please be mindful that the dataset locations for open-source LLMs and GPT may differ, so ensure to modify the 'root_path' accordingly.



