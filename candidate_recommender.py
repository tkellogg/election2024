from typing import Self, Dict, List, TypedDict
import json
import glob
import dspy
from duckduckgo_search import DDGS
import asyncio
import inspect
import os
from dotenv import load_dotenv
from anthropic import Anthropic
import time


# Load environment variables
load_dotenv()


class ClaudeLM(dspy.LM):
    """Claude language model configuration"""
    
    def __init__(self: Self):
        self.client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.model = "claude-3-sonnet-20240229"
        self.max_tokens = 4096
        super().__init__(model=self.model)

    def basic_request(self, prompt: str, **kwargs) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content


class CandidateInfo(TypedDict):
    name: str
    party: str
    research: str


class IssueAnalysisSignature(dspy.Signature):
    """Signature for identifying key issues in a race"""
    race = dspy.InputField(desc="The specific election race being analyzed")
    candidates = dspy.InputField(desc="List of candidates and their information")
    voter_preferences = dspy.InputField(desc="The voter's stated preferences and priorities")
    
    key_issues = dspy.OutputField(desc="List of the top 3 key issues for this race")
    issue_analysis = dspy.OutputField(desc="Detailed analysis of how the issues relate to voter preferences")


class RecommendationSignature(dspy.Signature):
    """Signature for final candidate recommendation"""
    candidates = dspy.InputField(desc="List of candidates and their information")
    preferences = dspy.InputField(desc="The voter's stated preferences")
    key_issues = dspy.InputField(desc="Key issues identified for this race")
    issue_analysis = dspy.InputField(desc="Analysis of the issues")
    
    recommendation = dspy.OutputField(desc="The recommended candidate")
    reasoning = dspy.OutputField(desc="Detailed reasoning for the recommendation")


class CandidateResearcher:
    def __init__(self: Self) -> None:
        # Initialize DSPy with Claude
        self.lm = ClaudeLM()
        dspy.settings.configure(lm=self.lm)
        
        # Initialize predictors with signatures
        self.issue_analyzer = dspy.ChainOfThought(IssueAnalysisSignature)
        self.recommender = dspy.ChainOfThought(RecommendationSignature)
        self.search_client = DDGS()

    async def search_candidate(self: Self, name: str, race: str) -> str:
        """Search for candidate information using DuckDuckGo"""
        results = self.search_client.text(
            f"North Carolina {name} {race} politician positions views", 
            region="us-en",
            max_results=3
        )
        return " ".join([result['body'] for result in results])

    async def analyze_race_issues(self: Self, race: str, candidates: List[Dict], preferences: str) -> dspy.Prediction:
        """Analyze key issues for a specific race"""
        candidate_info = []
        
        for candidate in candidates:
            info = await self.search_candidate(candidate['candidates'], race)
            candidate_info.append({
                'name': candidate['candidates'],
                'party': candidate['party'],
                'research': info
            })
        
        return self.issue_analyzer(race=race, candidates=candidate_info, research=preferences, voter_preferences=preferences)

    async def analyze_candidates(self: Self, race: str, candidates: List[Dict], preferences: str) -> dspy.Prediction:
        """Analyze candidates against user preferences"""
        # First analyze key issues
        issues_result = await self.analyze_race_issues(race, candidates, preferences)
        
        # Gather detailed candidate info
        candidate_info = []
        for candidate in candidates:
            info = await self.search_candidate(candidate['candidates'], race)
            candidate_info.append({
                'name': candidate['candidates'],
                'party': candidate['party'],
                'research': info
            })
        
        # Make final recommendation
        return self.recommender(
            candidates=candidate_info,
            preferences=preferences,
            key_issues=issues_result.key_issues,
            issue_analysis=issues_result.issue_analysis
        )


class RecommendationEngine:
    def __init__(self: Self, data_paths: List[str] | None = None):
        if data_paths is None:
            data_paths = glob.glob("*.json")
        self.data_paths = data_paths
        self.researcher = CandidateResearcher()

    def load_candidates(self: Self) -> Dict:
        """Load candidate data from all JSON files"""
        all_candidates = {}
        for path in self.data_paths:
            try:
                with open(path, 'r') as f:
                    candidates = json.load(f)
                    # Merge new candidates with existing ones
                    all_candidates.update(candidates)
            except json.JSONDecodeError:
                print(f"Warning: Skipping {path} - not a valid JSON file")
            except Exception as e:
                print(f"Warning: Could not load {path} - {str(e)}")
        return all_candidates

    async def get_recommendation(self: Self, race: str, preferences: str) -> dspy.Prediction:
        """Get candidate recommendation for a specific race"""
        all_candidates = self.load_candidates()
        if race not in all_candidates:
            raise ValueError(f"Race '{race}' not found in candidate data")
            
        return await self.researcher.analyze_candidates(race, all_candidates[race], preferences)


async def main():
    preferences = inspect.cleandoc("""
    """)
    
    engine = RecommendationEngine()
    processed_races = set()
    
    while True:
        # Load available races
        all_races = engine.load_candidates().keys()
        available_races = [race for race in all_races if race not in processed_races]
        
        if not available_races:
            print("\nAll races have been analyzed!")
            break
            
        print("\nAvailable races:")
        for idx, race in enumerate(available_races, 1):
            print(f"{idx}. {race}")
        print("\nEnter the number of the race you'd like to analyze (or 'q' to quit):")
        
        choice = input().strip()
        if choice.lower() == 'q':
            break
            
        try:
            race_idx = int(choice) - 1
            if 0 <= race_idx < len(available_races):
                selected_race = available_races[race_idx]
                print(f"\nAnalyzing {selected_race}...")
                
                start_time = time.time()
                recommendation = await engine.get_recommendation(selected_race, preferences)
                end_time = time.time()
                
                print(f"\nAnalysis completed in {end_time - start_time:.2f} seconds")
                print(f"Recommendation:\n{recommendation.completions[0]['recommendation']}")
                print(f"Reasoning:\n{recommendation.completions[0]['reasoning']}")
                print("\nPress 'Enter' to continue...")
                input()
                
                processed_races.add(selected_race)
            else:
                print("Invalid number. Please try again.")
        except ValueError as e:
            print(f"ValueError: {e}; input={choice}")
            print("Please enter a valid number or 'q' to quit.")


if __name__ == "__main__":
    asyncio.run(main())