from typing import List, Set
import re


class ContentFilter:
    def __init__(self):
        # Social media platforms to filter (excluding OnlyFans)
        self.social_media_terms: Set[str] = {
            'facebook', 'fb', 'instagram', 'ig', 'insta',
            'twitter', 'tweet', 'x.com', 'tiktok', 'snapchat',
            'snap', 'reddit', 'telegram', 'whatsapp', 'discord',
            'linkedin', 'pinterest', 'youtube', 'tumblr'
        }

        # Terms suggesting in-person meetings
        self.meeting_terms: Set[str] = {
            'meet up', 'meetup', 'meet in person', 'get together',
            'coffee', 'dinner', 'lunch', 'drinks', 'hang out',
            'hangout', 'see you in', 'meet you at', 'come over',
            'my place', 'your place', 'address', 'location',
            'where are you located', 'what city'
        }

        # Compile regex patterns
        self._social_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(term) for term in self.social_media_terms) + r')\b',
            re.IGNORECASE
        )

        self._meeting_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(term) for term in self.meeting_terms) + r')\b',
            re.IGNORECASE
        )

    def check_message(self, message: str) -> tuple[bool, List[str]]:
        violations = []

        social_matches = self._social_pattern.findall(message)
        if social_matches:
            violations.append(f"Contains social media references: {', '.join(set(social_matches))}")

        meeting_matches = self._meeting_pattern.findall(message)
        if meeting_matches:
            violations.append(f"Contains in-person meeting suggestions: {', '.join(set(meeting_matches))}")

        return (len(violations) == 0, violations)

    def filter_message(self, message: str) -> str:
        message = self._social_pattern.sub('[FILTERED]', message)
        message = self._meeting_pattern.sub('[FILTERED]', message)
        return message

    def suggest_alternatives(self, message: str) -> List[str]:
        suggestions = []

        if self._social_pattern.search(message):
            suggestions.append(
                "Instead of referring to other platforms, try focusing on OnlyFans features: "
                "'Check out my latest content here' or 'Send me a message on OnlyFans'"
            )

        if self._meeting_pattern.search(message):
            suggestions.append(
                "Instead of suggesting meetings, try: "
                "'Let's chat more here' or 'I love connecting with you through my content'"
            )

        return suggestions
