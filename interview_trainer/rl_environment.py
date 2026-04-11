import os
import time
from google import genai
from google.genai import types
from google.genai import errors

from dotenv import load_dotenv
load_dotenv()
# Setup Gemini
API_KEY = os.environ.get("GEMINI_API_KEY")
if API_KEY:
    client = genai.Client(api_key=API_KEY)
    print("Loaded API KEY successfully.")
else:
    client = genai.Client()
    print("WARNING: GEMINI_API_KEY not found in environment.")

def generate_content_with_retry(client, model, contents, config=None, max_retries=4):
    delay = 15
    for attempt in range(max_retries):
        try:
            if config:
                return client.models.generate_content(model=model, contents=contents, config=config)
            else:
                return client.models.generate_content(model=model, contents=contents)
        except errors.ClientError as e:
            if "API key not valid" in str(e):
                print("\n[Error] The provided GEMINI_API_KEY is invalid or has been revoked.")
                raise e
            elif "429" in str(e) or "quota" in str(e).lower() or "retryDelay" in str(e):
                if attempt == max_retries - 1:
                    print("\n[Error] API Rate limit / Quota exceeded. Please wait or check your Google Cloud billing.")
                    raise e
            else:
                raise e
                
            print(f"\n[API Rate Limit] Retry {attempt + 1}/{max_retries-1} in {delay}s...")
            time.sleep(delay)
            delay *= 2
        except errors.ServerError as e:
            if attempt == max_retries - 1:
                raise e
            print(f"\n[API Server Error] Retry {attempt + 1}/{max_retries-1} in {delay}s...")
            time.sleep(delay)
            delay *= 2

class StepResult:
    def __init__(self, attempt, answer, llm_score):
        self.attempt = attempt
        self.answer = answer
        self.nlp_score = 0
        self.llm_score = llm_score
        self.reward = llm_score
        self.delta = 0
        self.nlp_feedback = []
        self.llm_feedback = "Evaluated using Gemini"
        self.improvement_tip = "Improve based on feedback"

class EpisodeResult:
    def __init__(self, question, topic, steps):
        self.question = question
        self.topic = topic
        self.steps = steps
        self.best_reward = max(s.reward for s in steps)
        self.best_attempt = max(steps, key=lambda s: s.reward).attempt

class InterviewEnv:
    def __init__(self, question, topic, n_attempts=3, verbose=True):
        self.question = question
        self.topic = topic
        self.n_attempts = n_attempts
        self.verbose = verbose
        self.reward_engine = type('', (), {"max_reward": 10})()

    def generate_answer(self, previous_feedback=None):
        if previous_feedback:
            prompt = f"""
            You are a candidate in an interview. You previously answered an interview question but received the following feedback from the evaluator:

            {previous_feedback}

            Please heavily improve your answer to the following question using the feedback. Ensure your new answer uses significantly different phrasing, a completely fresh structure, and a new perspective from your previous attempts so it stands out as uniquely different:
            {self.question}
            """
        else:
            prompt = f"""
            Answer this interview question clearly:

            {self.question}
            """

        response = generate_content_with_retry(
            client,
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=1.0)
        )
        return response.text

    def llm_score(self, answer):
        prompt = f"""
        Evaluate this answer.

        Question: {self.question}
        Answer: {answer}

        Provide a score from 1 to 10 on the first line in the format: "Score: [number]".
        On the remaining lines, provide a short 1-2 sentence constructive feedback on how to improve.
        """

        response = generate_content_with_retry(
            client,
            model="gemini-2.5-flash",
            contents=prompt
        )
        text = response.text

        # Parse text
        import re
        score_match = re.search(r"Score:\s*(\d+)", text, re.IGNORECASE)
        number = int(score_match.group(1)) if score_match else 7

        feedback = text.replace(f"Score: {number}", "").strip()
        if score_match:
            lines = text.split("\n", 1)
            feedback = lines[-1].strip() if len(lines) > 1 else "Good job."

        return number, feedback

    def run_episode(self):
        steps = []
        feedback = None

        for i in range(1, self.n_attempts + 1):
            answer = self.generate_answer(previous_feedback=feedback)
            score, feedback = self.llm_score(answer)

            step = StepResult(i, answer, score)
            step.llm_feedback = feedback

            if steps:
                step.delta = step.reward - steps[-1].reward

            steps.append(step)

        return EpisodeResult(self.question, self.topic, steps)