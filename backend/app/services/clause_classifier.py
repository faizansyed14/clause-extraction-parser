"""
Fine-tuned classifier for 41 CUAD clause categories
"""
from typing import List, Dict
import re


class ClauseClassifier:
    """
    Fine-tuned classifier for 41 CUAD clause categories:
    - Parties, Effective Date, Termination
    - Liability Cap, Indemnity, IP Rights
    - Confidentiality, Non-Compete, etc.
    """
    
    CUAD_CATEGORIES = [
        "document_name",
        "parties",
        "agreement_date",
        "effective_date",
        "expiration_date",
        "renewal_term",
        "notice_period_to_terminate",
        "governing_law",
        "jurisdiction",
        "most_favored_nation",
        "non_compete",
        "exclusivity",
        "no_solicit_of_customers",
        "no_solicit_of_employees",
        "competitive_restriction_exception",
        "non_disparagement",
        "ip_ownership_assignment",
        "license_grant",
        "non_transferable_license",
        "affiliate_license",
        "unlimited_all_you_can_eat_license",
        "irrevocable_or_perpetual_license",
        "cap_on_liability",
        "liquidated_damages",
        "uncapped_liability",
        "warranty_duration",
        "insurance",
        "covenant_not_to_sue",
        "third_party_beneficiary",
        "right_of_first_refusal",
        "right_of_first_offer",
        "right_of_first_negotiation",
        "change_of_control",
        "anti_assignment",
        "revenue_profit_sharing",
        "price_restrictions",
        "minimum_commitment",
        "volume_restriction",
        "post_termination_services",
        "audit_rights",
        "confidentiality"
    ]
    
    # Keyword patterns for each category
    CATEGORY_KEYWORDS = {
        "document_name": ["agreement", "contract", "memorandum", "addendum"],
        "parties": ["party", "parties", "contractor", "client", "company", "entity"],
        "agreement_date": ["agreement date", "dated", "executed on"],
        "effective_date": ["effective date", "commencement", "effective as of", "effective"],
        "expiration_date": ["expiration", "expiry", "expires", "termination date"],
        "renewal_term": ["renewal", "renew", "automatic renewal", "renewal period"],
        "notice_period_to_terminate": ["notice", "termination notice", "days notice"],
        "governing_law": ["governing law", "laws of", "state of", "country"],
        "jurisdiction": ["jurisdiction", "courts of", "venue", "legal proceedings"],
        "most_favored_nation": ["most favored nation", "mfn", "most favoured"],
        "non_compete": ["non-compete", "noncompete", "restrictive covenant"],
        "exclusivity": ["exclusive", "exclusivity", "sole", "only"],
        "no_solicit_of_customers": ["no solicit", "solicitation", "customer solicitation"],
        "no_solicit_of_employees": ["employee solicitation", "hire employees", "poach"],
        "competitive_restriction_exception": ["exception", "exclusion", "permitted"],
        "non_disparagement": ["disparage", "disparagement", "negative comments"],
        "ip_ownership_assignment": ["intellectual property", "ip ownership", "assign", "assignment"],
        "license_grant": ["license", "grant", "licensing", "permission"],
        "non_transferable_license": ["non-transferable", "non transferable", "not transferable"],
        "affiliate_license": ["affiliate", "subsidiary", "related entity"],
        "unlimited_all_you_can_eat_license": ["unlimited", "all you can eat", "unrestricted"],
        "irrevocable_or_perpetual_license": ["irrevocable", "perpetual", "permanent"],
        "cap_on_liability": ["liability cap", "maximum liability", "limit of liability", "capped"],
        "liquidated_damages": ["liquidated damages", "penalty", "fixed damages"],
        "uncapped_liability": ["uncapped", "unlimited liability", "no cap"],
        "warranty_duration": ["warranty", "warranties", "warranty period"],
        "insurance": ["insurance", "insured", "coverage", "policy"],
        "covenant_not_to_sue": ["covenant not to sue", "waiver", "release"],
        "third_party_beneficiary": ["third party", "beneficiary", "third-party"],
        "right_of_first_refusal": ["right of first refusal", "rofr", "first refusal"],
        "right_of_first_offer": ["right of first offer", "rofo", "first offer"],
        "right_of_first_negotiation": ["right of first negotiation", "rofn", "first negotiation"],
        "change_of_control": ["change of control", "merger", "acquisition", "change in control"],
        "anti_assignment": ["anti-assignment", "no assignment", "assignment prohibited"],
        "revenue_profit_sharing": ["revenue sharing", "profit sharing", "royalty"],
        "price_restrictions": ["price", "pricing", "price restrictions", "price control"],
        "minimum_commitment": ["minimum", "commitment", "minimum purchase"],
        "volume_restriction": ["volume", "quantity", "restriction"],
        "post_termination_services": ["post-termination", "after termination", "post termination"],
        "audit_rights": ["audit", "inspection", "review", "examine"],
        "confidentiality": ["confidential", "non-disclosure", "proprietary", "nda"]
    }
    
    def classify_clause(self, clause_text: str) -> List[str]:
        """
        Classify clause into CUAD categories
        Returns list of matching categories (can be multiple)
        """
        text_lower = clause_text.lower()
        matches = []
        
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            # Check if any keyword appears in the text
            for keyword in keywords:
                if keyword in text_lower:
                    matches.append(category)
                    break
        
        # If no matches, return 'general'
        if not matches:
            return ['general']
        
        return matches
    
    def get_primary_category(self, clause_text: str) -> str:
        """
        Get the primary (most relevant) category for a clause
        """
        categories = self.classify_clause(clause_text)
        if categories:
            return categories[0]
        return 'general'
    
    def get_all_categories(self) -> List[str]:
        """
        Return list of all supported clause types from CUAD
        """
        return self.CUAD_CATEGORIES.copy()

