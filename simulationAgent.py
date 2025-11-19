from websocietysimulator import Simulator
from websocietysimulator.agent import SimulationAgent
from websocietysimulator.llm import LLMBase, InfinigenceLLM
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator.agent.modules.memory_modules import MemoryDILU
import json
import time

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
        
        # 定义商家类别及其关键评价维度
        self.category_dimensions = {
            'Restaurant': {
                'keywords': ['restaurant', 'food', 'cafe', 'coffee', 'bar', 'pub', 'dining', 'pizza', 'chinese', 'mexican', 'italian', 'burger', 'sandwich', 'bakery', 'breakfast', 'lunch', 'dinner'],
                'focus_areas': ['food quality', 'taste', 'portion size', 'menu variety', 'service speed', 'staff friendliness', 'ambiance', 'cleanliness', 'price-value ratio', 'wait time']
            },
            'Shopping': {
                'keywords': ['shopping', 'store', 'mall', 'boutique', 'retail', 'shop', 'market', 'grocery'],
                'focus_areas': ['product variety', 'product quality', 'pricing', 'staff helpfulness', 'store layout', 'checkout speed', 'return policy', 'cleanliness']
            },
            'Services': {
                'keywords': ['salon', 'beauty', 'spa', 'gym', 'fitness', 'hair', 'nail', 'massage', 'car wash', 'auto'],
                'focus_areas': ['service quality', 'staff professionalism', 'results', 'cleanliness', 'appointment scheduling', 'price-value ratio', 'facility conditions']
            },
            'Travel': {
                'keywords': ['hotel', 'motel', 'hostel', 'resort', 'inn', 'accommodation', 'lodging', 'attraction', 'tour'],
                'focus_areas': ['location', 'cleanliness', 'room conditions', 'facilities', 'staff service', 'check-in/out experience', 'amenities', 'value for money', 'noise level']
            },
            'Health': {
                'keywords': ['dentist', 'doctor', 'medical', 'clinic', 'health', 'dental', 'physician', 'hospital'],
                'focus_areas': ['professional expertise', 'wait time', 'staff courtesy', 'cleanliness', 'appointment availability', 'insurance handling', 'explanation clarity', 'treatment results']
            },
            'Others': {
                'keywords': ['education', 'school', 'pet', 'veterinary', 'service', 'repair', 'home'],
                'focus_areas': ['service quality', 'professionalism', 'price', 'results', 'customer service']
            }
        }
    
    def detect_item_category(self, item_info) -> str:
        """
        根据商家信息检测其类别
        
        :param item_info: 商家信息字典
        :return: 检测到的类别名称
        """
        categories = item_info.get('categories', '').lower()
        name = item_info.get('name', '').lower()
        
        combined_text = f"{categories} {name}"
        
        # 计算每个类别的匹配分数
        category_scores = {}
        for category, config in self.category_dimensions.items():
            score = sum(1 for keyword in config['keywords'] if keyword in combined_text)
            category_scores[category] = score
        
        # 返回得分最高的类别,如果都是0则返回"Others"
        max_category = max(category_scores.items(), key=lambda x: x[1])
        return max_category[0] if max_category[1] > 0 else 'Others'

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

    def generate_item_review_description(self, item_review_info, source: str, item_info: dict) -> str:
        """
        生成商家评论总结并返回响应。

        :param item_review_info: 包含商家评论信息的字典
        :param source: 数据源(例如 'yelp')
        :param item_info: 商家信息字典,用于检测类别
        :return: 模型生成的自然语言描述
        """
        # 检测商家类别
        category = self.detect_item_category(item_info)
        focus_areas = self.category_dimensions.get(category, {}).get('focus_areas', [])
        focus_areas_str = ', '.join(focus_areas)

        # 将 item_review_info 转换为字符串
        item_review_info_str = str(item_review_info)

        # 创建提示词,根据类别动态调整
        prompt_filled = f"""The following is a collection of user reviews for a {category} business from the {source} dataset. 
        Please write a comprehensive summary of the reviews in natural language. 
        Your description should:
        1. Summarize the general sentiment (positive or negative) expressed across the reviews.
        2. Highlight recurring themes or issues particularly related to: {focus_areas_str}.
        3. Mention any specific strengths or weaknesses pointed out by multiple reviewers.
        4. If applicable, include the most common ratings or complaints.
        5. Avoid repeating the same information and keep the summary balanced and fair, reflecting both positive and negative feedback.
        6. Focus on the key aspects that matter most for this type of business.
        Please ensure the review summary is clear, concise, and informative, with a word limit of around 300 words:
        {item_review_info_str}"""
        
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
            item_reviews_text = self.generate_item_review_description(item_reviews, source, item_info)
            user_reviews_text = self.generate_user_review_description(user_reviews, source)

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
    number_of_tasks = 10
    simulator = Simulator(data_dir="./dataset", device="cpu", cache=True)
    simulator.set_task_and_groundtruth(task_dir=f"./example/track1/{task_set}/tasks",
                                       groundtruth_dir=f"./example/track1/{task_set}/groundtruth")

    # Set the agent and LLM
    simulator.set_agent(MySimulationAgent)
    # Set LLM API key here
    simulator.set_llm(InfinigenceLLM(api_key="sk-f3obch3f7dfxlg4l"))

    # Run the simulation
    # If you don't set the number of tasks, the simulator will run all tasks.
    outputs = simulator.run_simulation(number_of_tasks, enable_threading=True, max_workers=5)
    
    # Evaluate the agent
    evaluation_results = simulator.evaluate()
    evaluation_history = simulator.get_evaluation_history()

    task_info = []
    
    # Save each output with task, groundtruth, user_info, item_info
    for idx, output in enumerate(outputs):

        task = simulator.tasks[idx].to_dict() if hasattr(simulator.tasks[idx], 'to_dict') else simulator.tasks[idx]
        groundtruth = simulator.groundtruth_data[idx]
        user_id = task.get('user_id')
        item_id = task.get('item_id')
        user_info = simulator.interaction_tool.get_user(user_id) if user_id else None
        item_info = simulator.interaction_tool.get_item(item_id) if item_id else None
        # 组织保存内容
        task_info.append({
            'task': task,
            'groundtruth': groundtruth,
            'user_info': user_info,
            'item_info': item_info,
            'model_output': output['output'] if isinstance(output, dict) and 'output' in output else output
        })

    evaluation_results["task_info"] = task_info
    # evaluation_results["evaluation_history"] = evaluation_history

    with open(f'./evaluation_results/{task_set}_track1_{number_of_tasks}tasks_{time.time()}.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)