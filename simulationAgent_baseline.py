from websocietysimulator import Simulator
from websocietysimulator.agent import SimulationAgent
from websocietysimulator.llm import LLMBase, InfinigenceLLM
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator.agent.modules.memory_modules import MemoryDILU
import json

class PlanningBaseline(PlanningBase):
    """Inherit from PlanningBase"""

    def __init__(self, llm):
        """Initialize the planning module"""
        super().__init__(llm=llm)

    def __call__(self, task_description):
        """Override the parent class's __call__ method"""
        self.plan = [
            {
                'description': 'First I need to find user information',
                'reasoning instruction': 'None',
                'tool use instruction': {task_description['user_id']}
            },
            {
                'description': 'Next, I need to find business information',
                'reasoning instruction': 'None',
                'tool use instruction': {task_description['item_id']}
            }
        ]
        return self.plan


class ReasoningBaseline(ReasoningBase):
    """Inherit from ReasoningBase"""

    def __init__(self, profile_type_prompt, llm):
        """Initialize the reasoning module"""
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)

    def __call__(self, task_description: str):
        """Override the parent class's __call__ method"""

        messages = [{"role": "user", "content": task_description}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.0,
            max_tokens=1000
        )

        return reasoning_result


class MySimulationAgent(SimulationAgent):
    """Participant's implementation of SimulationAgent."""

    def __init__(self, llm: LLMBase):
        """Initialize MySimulationAgent"""
        super().__init__(llm=llm)
        self.planning = PlanningBaseline(llm=self.llm)
        self.reasoning = ReasoningBaseline(profile_type_prompt='', llm=self.llm)
        self.memory = MemoryDILU(llm=self.llm)

    def generate_user_description(self, user_info, source) -> str:
        """
        生成用户描述并返回响应。

        :param user_info: 包含用户信息的字典
        :param source: 数据源（例如 'yelp'）
        """

        # 将 user_info 转换为字符串
        user_info_str = str(user_info)

        # 创建提示词
        prompt_filled = f"""The following is a user data of the {source} dataset after processing. Please write a paragraph in natural language describing the user. The description should cover their review activity (how many reviews they have written and their impact on the community, such as "useful," "funny," and "cool" votes), their elite status (mention the years they were considered Elite), and any notable aspects of their social circle (such as the number of friends they have). Make sure to convey the user's personality and activity on the platform, without including raw data, and keep it in a conversational tone,Your description should be fair and just,Keep your reply to around 100 words,Please describe in the second person:{user_info_str}"""

        # 调用模型生成响应
        response = self.reasoning(prompt_filled)

        return response

    def generate_item_description(self, item_info, source: str) -> str:
        """
        生成用户描述并返回响应。

        :param user_info: 包含用户信息的字典
        :param source: 数据源（例如 'yelp'）
        """

        # 将 user_info 转换为字符串
        item_info_str = str(item_info)
        """
        请用自然语言撰写一段约100字的描述，涵盖商家名称、所在地址（街道、城市和州）、业务类型、评分（星级）及评论数量。
        如有具体数字，请尽量体现。同时说明其是否提供外卖、外带、预订服务，以及是否配备电视或停车位。
        还需提及氛围、噪音水平和主要客群特征（如适合团体、休闲聚会等）。
        描述应客观公正，体现该商家的特色、顾客体验与整体氛围。
        """

        # 创建提示词
        prompt_filled = f"""The following is a business data of the {source} dataset after processing. 
Please write a paragraph in natural language describing the business. 
If there are numbers in it, it is best to describe them.
The description should cover the name of the business, its location (address, city, and state), the type of business, its rating (stars), and the number of reviews. 
Mention the attributes such as whether it offers delivery, takeout, reservations, and whether it has a TV or parking. 
Also include any notable features like the ambiance, noise level, and what type of crowd it attracts (e.g., good for groups, casual, etc.). 
Make sure to describe the business in a way that reflects its character, customer experience, and atmosphere. 
Your description should be fair and just, and keep your reply to around 100 words:
{item_info_str}"""

        # 调用模型生成响应
        response = self.reasoning(prompt_filled)
        return response

    def generate_item_review_description(self, item_review_info, source: str) -> str:
        """
        生成用户描述并返回响应。

        :param user_info: 包含用户信息的字典
        :param source: 数据源（例如 'yelp'）
        :return: 模型生成的自然语言描述
        """

        # 将 user_info 转换为字符串
        item_review_info = str(item_review_info)

        """
        以下内容是来自 {source} 数据集的该商家用户评论合集。
        请用自然语言撰写一篇全面的评论摘要。
        你的描述应满足以下要求：

        概括评论中普遍表达的情绪倾向（正面或负面）。
        突出反复出现的主题或问题（例如服务差、食物质量、氛围等）。
        提及多位评论者指出的具体优点或缺点（例如员工友善、上菜慢、菜品美味等）。
        如适用，请包含最常见的评分或投诉内容（例如平均评分、卫生或价格方面的问题）。
        避免重复相同信息，并保持摘要的平衡与公正，同时反映正面与负面反馈。
        请确保评论摘要清晰、简洁且信息丰富，字数控制在300字左右。
        """

        # 创建提示词
        prompt_filled = f"""The following is a collection of user reviews for the business  from the {source} dataset. 
        Please write a comprehensive summary of the reviews in natural language. 
        Your description should:
        1. Summarize the general sentiment (positive or negative) expressed across the reviews.
        2. Highlight recurring themes or issues (e.g., poor service, food quality, atmosphere).
        3. Mention any specific strengths or weaknesses pointed out by multiple reviewers (e.g., friendly staff, slow service, great food).
        4. If applicable, include the most common ratings or complaints (e.g., average rating, issues with cleanliness or pricing).
        5. Avoid repeating the same information and keep the summary balanced and fair, reflecting both positive and negative feedback.
        Please ensure the review summary is clear, concise, and informative, with a word limit of around 300 words:
        {item_review_info}"""
        # 调用模型生成响应
        response = self.reasoning(prompt_filled)

        return response

    def generate_user_review_description(self, user_review_info, source: str) -> str:
        """
        生成用户描述并返回响应。

        :param user_info: 包含用户信息的字典
        :param source: 数据源（例如 'yelp'）
        :return: 模型生成的自然语言描述
        """

        # 将 user_info 转换为字符串
        user_review_info = str(user_review_info)

        """
        以下是你在 {source} 数据集中撰写的历史评论合集。  
        请分析你的评论风格并总结其特点，重点关注以下方面：  
        1. 你的语气和用词（例如正式、随意、批判性、幽默等）；  
        2. 你经常评论的主题或内容（例如食物质量、服务、氛围等）；  
        3. 评论的详细程度（例如是否提供具体例子，还是较为笼统）；  
        4. 你的情绪倾向（例如客观平衡、过于积极或过于消极）；  
        5. 评分中的任何规律（例如倾向于给中等分数、极端分数，或评分一贯稳定）；  
        6. 你如何评价体验中的优点与不足。  
        基于以上分析，请用第二人称写一段约100字的自然语言总结，突出你撰写评论的典型风格。
        """

        # 创建提示词
        prompt_filled = f"""The following is a collection of historical reviews written by a user from the {source} dataset. 
        Please analyze the user's reviewing style and summarize their characteristics. 
        Focus on the following aspects:
        1. The user's tone and language (e.g., formal, casual, critical, humorous).
        2. Common themes or topics the user frequently comments on (e.g., food quality, service, atmosphere).
        3. The level of detail in the reviews (e.g., whether they provide specific examples or remain general).
        4. The user's sentiment (e.g., balanced, overly positive, overly negative).
        5. Any patterns in the user's ratings (e.g., tends to give moderate scores, extremes, or consistent ratings).
        6. How the user evaluates positives and negatives in the experience.
        Based on this analysis, provide a concise summary of the user's reviewing style in natural language, highlighting their typical approach to writing reviews. Limit the summary to around 100 words,Please describe in the second person:
        {user_review_info}"""

        # 调用模型生成响应
        response = self.reasoning(prompt_filled)

        return response

    def workflow(self):
        """
        Simulate user behavior
        Returns:
            tuple: (star (float), useful (float), funny (float), cool (float), review_text (str))
        """
        try:
            user_id = self.task.get('user_id')
            item_id = self.task.get('item_id')

            # 获取相关评论
            item_reviews = self.interaction_tool.get_reviews(item_id=item_id)
            user_reviews = self.interaction_tool.get_reviews(user_id=user_id)
            user_info = self.interaction_tool.get_user(user_id=user_id)
            item_info = self.interaction_tool.get_item(item_id=item_id)
            # 随机选择一些评论以供参考
            # random_item_reviews = random.sample(item_reviews, min(len(item_reviews), 5))
            # random_user_reviews = random.sample(user_reviews, min(len(user_reviews), 5))

            item_reviews_text = ""
            user_reviews_text = ""

            # 获取用户来源信息
            source = user_info.get('source')
            after_user_info = self.generate_user_description(user_info, source)
            item_description = self.generate_item_description(item_info, source)
            item_reviews_text = self.generate_item_review_description(item_reviews, source)
            user_reviews_text = self.generate_user_review_description(user_reviews, source)

            """
            你是在 {source} 数据集上的一位真实人类用户，该平台是一个众包商业评论网站。以下是你的一些背景信息，包括你在平台上的部分活动：  
            {after_user_info}，{user_reviews_text}。  
            ###该商家的整体情况：{item_description}###  
            ###以下是该商家过往收到的评论摘要：{item_reviews_text}###  

            你需要为这家商家撰写一条评论，请仔细分析以下方面：  
            1. 根据你的用户画像和评论风格，你会给这家商家打多少分？请注意，许多用户在体验远超预期时会给出5星，而在体验极差、连基本标准都未达到时会给出1星。  
            2. 结合商家详情和你过去的经历，你会具体评论哪些方面？请聚焦于让这家商家脱颖而出的优点，或严重影响体验的缺点。  

            要求：  
            - 星级评分必须为以下之一：1.0、2.0、3.0、4.0、5.0  
            - 若商家在关键方面达到或超出预期，可考虑给5星  
            - 若商家在关键方面严重不足，可考虑给1星  
            - 评论正文应为2至4句话，聚焦你的个人体验和情感反应  
            - “有用/有趣/酷”计数应为非负整数，反映可能的用户互动程度  
            - 保持与你历史评论风格和评分习惯的一致性  
            - 聚焦商家的具体细节，而非泛泛而谈  
            - 当商家提供优质服务或产品时，请大方给高分  
            - 当商家未达基本标准时，请明确指出并给予低分  

            请严格按照以下格式输出你的回复：  
            stars: [你的评分]  
            review: [你的评论]
            """

            task_description = f'''
        You are a real human user on {source} dataset, a platform for crowd-sourced business reviews. Here is some description of your past, including some of your activities on the platform: 
        {after_user_info},{user_reviews_text}.
            ###The overall situation of this business: {item_description}###
            ###Here is a summary of the reviews this product has received in the past: {item_reviews_text}###
            You need to write a review for this business,
            Please analyze the following aspects carefully:
            1. Based on your user profile and review style, what rating would you give this business? Remember that many users give 5-star ratings for excellent experiences that exceed expectations, and 1-star ratings for very poor experiences that fail to meet basic standards.
            2. Given the business details and your past experiences, what specific aspects would you comment on? Focus on the positive aspects that make this business stand out or negative aspects that severely impact the experience.
            Requirements:
            - Star rating must be one of: 1.0, 2.0, 3.0, 4.0, 5.0
            - If the business meets or exceeds expectations in key areas, consider giving a 5-star rating
            - If the business fails significantly in key areas, consider giving a 1-star rating
            - Review text should be 2-4 sentences, focusing on your personal experience and emotional response
            - Useful/funny/cool counts should be non-negative integers that reflect likely user engagement
            - Maintain consistency with your historical review style and rating patterns
            - Focus on specific details about the business rather than generic comments
            - Be generous with ratings when businesses deliver quality service and products
            - Be critical when businesses fail to meet basic standards

            Format your response exactly as follows:
            stars: [your rating]
            review: [your review]
        '''
            result = self.reasoning(task_description)

            stars_line = [line for line in result.split('\n') if 'stars:' in line][0]
            review_line = [line for line in result.split('\n') if 'review:' in line][0]

            stars = float(stars_line.split(':')[1].strip())
            review_text = review_line.split(':')[1].strip()

            if len(review_text) > 512:
                review_text = review_text[:512]

            return {
                "stars": stars,
                "review": review_text
            }
        except Exception as e:
            print(f"Error in workflow: {e}")
            return {
                "stars": 3,
                "review": ""
            }


if __name__ == "__main__":
    # Set the data
    task_set = "yelp"
    simulator = Simulator(data_dir="./dataset", device="gpu", cache=True)
    simulator.set_task_and_groundtruth(task_dir=f"./example/track1/{task_set}/tasks",
                                       groundtruth_dir=f"./example/track1/{task_set}/groundtruth")

    # Set the agent and LLM
    simulator.set_agent(MySimulationAgent)
    # Set LLM API key here
    simulator.set_llm(InfinigenceLLM(api_key="sk-f3obch3f7dfxlg4l"))

    # Run the simulation
    # If you don't set the number of tasks, the simulator will run all tasks.
    outputs = simulator.run_simulation(number_of_tasks=1, enable_threading=True, max_workers=5)

    # Evaluate the agent
    evaluation_results = simulator.evaluate()
    with open(f'./evaluation_results_track1_{task_set}.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    # Get evaluation history
    evaluation_history = simulator.get_evaluation_history()