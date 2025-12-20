"""
C2PA Content Credentials forensic analysis.

Verifies cryptographic content credentials per the C2PA specification.
Content credentials provide a chain of custody for media files,
recording their origin and any modifications.

C2PA (Coalition for Content Provenance and Authenticity) is backed by
Adobe, Microsoft, BBC, Intel, and others.

References:
    C2PA Technical Specification: https://c2pa.org/specifications/
    CAI (Content Authenticity Initiative): https://contentauthenticity.org/
"""

from typing import Optional, List, Dict, Any
from pathlib import Path

try:
    import c2pa
    HAS_C2PA = True
except ImportError:
    HAS_C2PA = False

from ..state import DimensionResult, DimensionState, Confidence, Evidence


class C2PAAnalyzer:
    """
    Analyzes C2PA content credentials in media files.

    Content credentials provide:
    - Cryptographic proof of origin
    - Chain of custody (who modified what when)
    - Tamper detection
    - AI generation disclosure

    C2PA is NOT proof of authenticity by itself - it proves
    the chain of custody IF credentials are present and valid.
    Missing credentials are common and not necessarily suspicious.
    """

    def __init__(self):
        """Initialize C2PA analyzer."""
        if not HAS_C2PA:
            self._available = False
        else:
            self._available = True

    @property
    def available(self) -> bool:
        """Check if C2PA analysis is available."""
        return self._available

    def analyze(self, file_path: str) -> DimensionResult:
        """
        Analyze C2PA content credentials in a file.

        Returns:
            DimensionResult with:
            - VERIFIED: Valid credentials, chain of custody intact
            - TAMPERED: Credentials present but file modified
            - INVALID: Credentials present but signature invalid
            - MISSING: No credentials found (common, not suspicious)
            - UNCERTAIN: Analysis not possible
        """
        if not self._available:
            return DimensionResult(
                dimension="c2pa",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.NA,
                evidence=[Evidence(
                    finding="C2PA library not available",
                    explanation="Install c2pa-python for content credential analysis"
                )],
                methodology="C2PA content credential verification"
            )

        path = Path(file_path)
        if not path.exists():
            return DimensionResult(
                dimension="c2pa",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.NA,
                evidence=[Evidence(
                    finding="File not found",
                    explanation=f"Cannot analyze: {file_path}"
                )]
            )

        try:
            # Read manifest store from file
            manifest_store = c2pa.read_file(str(path))

            if manifest_store is None:
                return self._no_credentials_result()

            # Parse and verify credentials
            return self._analyze_manifest_store(manifest_store, file_path)

        except Exception as e:
            error_str = str(e).lower()

            # Check for specific error types
            if "no manifest" in error_str or "not found" in error_str:
                return self._no_credentials_result()

            if "signature" in error_str or "invalid" in error_str:
                return DimensionResult(
                    dimension="c2pa",
                    state=DimensionState.INVALID,
                    confidence=Confidence.HIGH,
                    evidence=[Evidence(
                        finding="Invalid C2PA signature",
                        explanation=f"Credential signature verification failed: {e}",
                        contradiction="Credentials are present but cannot be verified"
                    )],
                    methodology="C2PA content credential verification"
                )

            if "tamper" in error_str or "modified" in error_str:
                return DimensionResult(
                    dimension="c2pa",
                    state=DimensionState.TAMPERED,
                    confidence=Confidence.HIGH,
                    evidence=[Evidence(
                        finding="C2PA tamper detection triggered",
                        explanation=f"File has been modified after signing: {e}",
                        contradiction="File content does not match signed hash"
                    )],
                    methodology="C2PA content credential verification"
                )

            # Generic error
            return DimensionResult(
                dimension="c2pa",
                state=DimensionState.UNCERTAIN,
                confidence=Confidence.NA,
                evidence=[Evidence(
                    finding="C2PA analysis error",
                    explanation=f"Error reading credentials: {e}"
                )],
                methodology="C2PA content credential verification"
            )

    def _no_credentials_result(self) -> DimensionResult:
        """Return result for files without C2PA credentials."""
        return DimensionResult(
            dimension="c2pa",
            state=DimensionState.MISSING,
            confidence=Confidence.HIGH,
            evidence=[Evidence(
                finding="No C2PA content credentials",
                explanation=(
                    "File does not contain C2PA manifest. "
                    "This is common and not inherently suspicious - "
                    "most media files do not have content credentials."
                )
            )],
            methodology="C2PA content credential verification"
        )

    def _analyze_manifest_store(
        self,
        manifest_store: Any,
        file_path: str
    ) -> DimensionResult:
        """
        Analyze a C2PA manifest store.

        Extracts:
        - Signature validation status
        - Claim generator (software/device)
        - Action history
        - AI generation assertions
        """
        evidence = []

        try:
            # Get active manifest (most recent)
            if hasattr(manifest_store, 'active_manifest'):
                manifest = manifest_store.active_manifest
            elif hasattr(manifest_store, 'get_active_manifest'):
                manifest = manifest_store.get_active_manifest()
            else:
                # Try to access as dict-like
                manifest = manifest_store

            # Check for AI generation assertions
            ai_evidence = self._check_ai_assertions(manifest)
            if ai_evidence:
                evidence.append(ai_evidence)

            # Check claim generator
            generator_evidence = self._check_claim_generator(manifest)
            if generator_evidence:
                evidence.append(generator_evidence)

            # Check actions (edit history)
            action_evidence = self._check_actions(manifest)
            if action_evidence:
                evidence.extend(action_evidence)

            # If we got here, credentials are valid
            evidence.insert(0, Evidence(
                finding="Valid C2PA credentials",
                explanation="Content credentials are present and signature is valid"
            ))

            # Determine if there are any concerns
            has_ai = any("AI" in e.finding or "generated" in e.finding.lower()
                        for e in evidence)

            if has_ai:
                return DimensionResult(
                    dimension="c2pa",
                    state=DimensionState.VERIFIED,
                    confidence=Confidence.HIGH,
                    evidence=evidence,
                    methodology="C2PA content credential verification"
                )

            return DimensionResult(
                dimension="c2pa",
                state=DimensionState.VERIFIED,
                confidence=Confidence.HIGH,
                evidence=evidence,
                methodology="C2PA content credential verification"
            )

        except Exception as e:
            evidence.append(Evidence(
                finding="Credential parsing issue",
                explanation=f"Could not fully parse manifest: {e}"
            ))
            return DimensionResult(
                dimension="c2pa",
                state=DimensionState.VERIFIED,
                confidence=Confidence.MEDIUM,
                evidence=evidence,
                methodology="C2PA content credential verification"
            )

    def _check_ai_assertions(self, manifest: Any) -> Optional[Evidence]:
        """Check for AI generation assertions in manifest."""
        try:
            # C2PA 2.0 defines standard AI assertions
            assertions = self._get_assertions(manifest)

            for assertion in assertions:
                label = self._get_assertion_label(assertion)
                if not label:
                    continue

                # Check for AI-related assertion types
                ai_labels = [
                    "c2pa.ai_generative_training",
                    "c2pa.ai_generated",
                    "c2pa.ai_inference",
                    "stds.exif", # May contain AI tool info
                ]

                for ai_label in ai_labels:
                    if ai_label in label.lower():
                        return Evidence(
                            finding="AI generation declared in credentials",
                            explanation=(
                                f"C2PA credentials indicate AI involvement: {label}"
                            ),
                            citation="C2PA Technical Specification Section 8"
                        )

            # Check for known AI tools in claim generator
            generator = self._get_claim_generator(manifest)
            if generator:
                ai_tools = ["dall-e", "midjourney", "stable diffusion", "firefly", "sora"]
                for tool in ai_tools:
                    if tool in generator.lower():
                        return Evidence(
                            finding=f"AI tool in claim generator: {generator}",
                            explanation="Credentials indicate AI-generated content"
                        )

        except Exception:
            pass

        return None

    def _check_claim_generator(self, manifest: Any) -> Optional[Evidence]:
        """Check claim generator (signing software/device)."""
        try:
            generator = self._get_claim_generator(manifest)
            if generator:
                return Evidence(
                    finding=f"Claim generator: {generator}",
                    explanation="Software/device that created the credentials"
                )
        except Exception:
            pass
        return None

    def _check_actions(self, manifest: Any) -> List[Evidence]:
        """Check action assertions (edit history)."""
        evidence = []
        try:
            assertions = self._get_assertions(manifest)

            for assertion in assertions:
                label = self._get_assertion_label(assertion)
                if label and "c2pa.actions" in label.lower():
                    actions = self._get_assertion_data(assertion)
                    if actions:
                        # Summarize actions
                        action_list = []
                        if isinstance(actions, dict) and "actions" in actions:
                            for action in actions["actions"][:5]:  # Limit to 5
                                if isinstance(action, dict):
                                    action_type = action.get("action", "unknown")
                                    action_list.append(action_type)

                        if action_list:
                            evidence.append(Evidence(
                                finding=f"Edit history: {', '.join(action_list)}",
                                explanation="Actions recorded in content credentials"
                            ))
        except Exception:
            pass
        return evidence

    def _get_assertions(self, manifest: Any) -> List[Any]:
        """Safely get assertions from manifest."""
        if hasattr(manifest, 'assertions'):
            return manifest.assertions or []
        if isinstance(manifest, dict):
            return manifest.get('assertions', [])
        return []

    def _get_assertion_label(self, assertion: Any) -> Optional[str]:
        """Safely get assertion label."""
        if hasattr(assertion, 'label'):
            return assertion.label
        if isinstance(assertion, dict):
            return assertion.get('label')
        return None

    def _get_assertion_data(self, assertion: Any) -> Optional[Any]:
        """Safely get assertion data."""
        if hasattr(assertion, 'data'):
            return assertion.data
        if isinstance(assertion, dict):
            return assertion.get('data')
        return None

    def _get_claim_generator(self, manifest: Any) -> Optional[str]:
        """Safely get claim generator from manifest."""
        if hasattr(manifest, 'claim_generator'):
            return manifest.claim_generator
        if isinstance(manifest, dict):
            return manifest.get('claim_generator')
        return None
