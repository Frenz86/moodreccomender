import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import io
import os
from openai import OpenAI
from audio_recorder_streamlit import audio_recorder
import json
from typing import Dict, List, Any, Optional
#from dotenv import load_dotenv

#load_dotenv()
MODEL = "gpt-4o-mini"
openaikey = st.secrets["OPENAI_API_KEY"]

class AudioProcessor:
    """Handles speech recognition from audio input"""
    def process(self, audio_bytes: bytes) -> str:
        recognizer = sr.Recognizer()
        try:
            with io.BytesIO(audio_bytes) as audio_file:
                with sr.AudioFile(audio_file) as source:
                    audio = recognizer.record(source)
                text = recognizer.recognize_google(audio, language="it-IT")
                return text
        except Exception as e:
            st.error(f"Errore nel riconoscimento vocale: {str(e)}")
            return ""

class EmotionalAnalyzer:
    """Analyzes emotions from text input"""
    def __init__(self, client: OpenAI):
        self.client = client

    def analyze(self, text: str) -> Dict[str, Any]:
        try:
            prompt = """Analizza il testo e fornisci un'analisi dettagliata delle emozioni.
            Restituisci SOLO un JSON con questa struttura:
            {
                "emotional_state": {
                    "primary": {
                        "emotion": "nome emozione principale",
                        "intensity": "intensità da 1 a 100",
                        "description": "breve descrizione"
                    },
                    "secondary": {
                        "emotion": "nome emozione secondaria",
                        "intensity": "intensità da 1 a 100",
                        "description": "breve descrizione"
                    }
                },
                "musical_preferences": {
                    "suggested_tempo": "range BPM consigliato",
                    "mood": "mood musicale suggerito",
                    "genres": ["genere1", "genere2", "genere3"],
                    "characteristics": ["caratteristica1", "caratteristica2"]
                },
                "therapeutic_goals": {
                    "primary_goal": "obiettivo principale",
                    "approach": "approccio suggerito",
                    "duration": "durata consigliata della sessione",
                    "expected_benefits": ["beneficio1", "beneficio2"]
                }
            }"""

            response = self.client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )

            return json.loads(response.choices[0].message.content)
        except Exception as e:
            st.error(f"Errore nell'analisi emotiva: {str(e)}")
            return {}

class MusicRecommender:
    """Generates specific music recommendations based on emotional analysis"""
    def __init__(self, client: OpenAI):
        self.client = client

    def recommend(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            prompt = f"""Sei un esperto musicoterapeuta. Basandoti su questa analisi emotiva:
            {json.dumps(analysis, indent=2, ensure_ascii=False)}
            
            Genera un oggetto JSON con ESATTAMENTE questa struttura:
            {{
                "recommendations": [
                    {{
                        "title": "titolo REALE del brano",
                        "artist": "nome REALE dell'artista",
                        "album": "nome dell'album",
                        "year": anno di uscita,
                        "genre": "genere principale",
                        "subgenre": "sottogenere specifico",
                        "musical_features": {{
                            "bpm": "velocità PRECISA del brano",
                            "key": "tonalità",
                            "energy": "livello energia 1-10",
                            "mood": "mood dominante"
                        }},
                        "therapeutic_value": {{
                            "primary_effect": "effetto principale",
                            "emotional_impact": "impatto emotivo previsto",
                            "listening_setting": "ambiente consigliato",
                            "best_moment": "momento ideale di ascolto"
                        }},
                        "reason": "spiegazione dettagliata della scelta"
                    }}
                ]
            }}
            
            REQUISITI FONDAMENTALI:
            - Raccomanda SOLO brani realmente esistenti
            - Includi ESATTAMENTE 5 brani diversi
            - Scegli brani specifici e rilevanti per lo stato emotivo
            - Varia generi e periodi storici
            - Considera il contesto culturale italiano
            - Fornisci dettagli precisi e verificabili
            """

            response = self.client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "Genera raccomandazioni musicali dettagliate"}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            recommendations = result.get("recommendations", [])

            if not recommendations:
                return self._get_default_recommendations()

            return recommendations

        except Exception as e:
            print(f"Errore nelle raccomandazioni: {str(e)}")
            return self._get_default_recommendations()

    def _get_default_recommendations(self) -> List[Dict[str, Any]]:
        return [
            {
                "title": "Nel blu dipinto di blu (Volare)",
                "artist": "Domenico Modugno",
                "album": "Nel blu dipinto di blu",
                "year": 1958,
                "genre": "Canzone Italiana",
                "subgenre": "Musica Leggera",
                "musical_features": {
                    "bpm": "125",
                    "key": "D Major",
                    "energy": 8,
                    "mood": "Gioioso"
                },
                "therapeutic_value": {
                    "primary_effect": "Elevazione del morale",
                    "emotional_impact": "Senso di libertà e leggerezza",
                    "listening_setting": "Ambienti aperti e luminosi",
                    "best_moment": "Mattina o primo pomeriggio"
                },
                "reason": "Classico che trasmette gioia e spensieratezza"
            }
        ] * 3

class AudioRenderer:
    """Handles text-to-speech conversion"""
    @staticmethod
    def render(text: str) -> Optional[bytes]:
        try:
            tts = gTTS(text=text, lang='it')
            audio_fp = io.BytesIO()
            tts.write_to_fp(audio_fp)
            audio_fp.seek(0)
            return audio_fp
        except Exception as e:
            st.error(f"Errore nella sintesi vocale: {str(e)}")
            return None

class ResponseFormatter:
    """Formats all responses for display"""
    @staticmethod
    def format_analysis(analysis: Dict[str, Any]) -> str:
        try:
            text = "🎭 ANALISI EMOTIVA\n"
            text += "═══════════════\n\n"
            
            emotional = analysis.get("emotional_state", {})
            primary = emotional.get("primary", {})
            secondary = emotional.get("secondary", {})
            
            text += f"📌 Emozione Principale: {primary.get('emotion', 'N/A')}\n"
            text += f"   Intensità: {primary.get('intensity', 'N/A')}/100\n"
            text += f"   {primary.get('description', '')}\n\n"
            
            text += f"📌 Emozione Secondaria: {secondary.get('emotion', 'N/A')}\n"
            text += f"   Intensità: {secondary.get('intensity', 'N/A')}/100\n"
            text += f"   {secondary.get('description', '')}\n\n"
            
            musical = analysis.get("musical_preferences", {})
            text += "🎵 PREFERENZE MUSICALI\n"
            text += "═══════════════════\n"
            text += f"• Tempo: {musical.get('suggested_tempo', 'N/A')}\n"
            text += f"• Mood: {musical.get('mood', 'N/A')}\n"
            text += f"• Generi: {', '.join(musical.get('genres', ['N/A']))}\n\n"
            
            therapeutic = analysis.get("therapeutic_goals", {})
            text += "🎯 OBIETTIVI TERAPEUTICI\n"
            text += "═══════════════════════\n"
            text += f"• {therapeutic.get('primary_goal', 'N/A')}\n"
            text += f"• Durata: {therapeutic.get('duration', 'N/A')}\n"
            text += f"• Benefici attesi: {', '.join(therapeutic.get('expected_benefits', ['N/A']))}\n"
            
            return text
        except Exception as e:
            return f"Errore nella formattazione dell'analisi: {str(e)}"

    @staticmethod
    def format_recommendations(recommendations: List[Dict[str, Any]]) -> str:
        try:
            text = "🎵 BRANI CONSIGLIATI\n"
            text += "════════════════\n\n"
            
            for i, song in enumerate(recommendations, 1):
                text += f"{i}. {song.get('artist', 'N/A')} - {song.get('title', 'N/A')}\n"
                text += f"   📀 Album: {song.get('album', 'N/A')} ({song.get('year', 'N/A')})\n"
                text += f"   🎼 Genere: {song.get('genre', 'N/A')} / {song.get('subgenre', 'N/A')}\n"
                
                features = song.get('musical_features', {})
                text += f"   🎹 BPM: {features.get('bpm', 'N/A')} | "
                text += f"Tonalità: {features.get('key', 'N/A')} | "
                text += f"Energia: {features.get('energy', 'N/A')}/10\n"
                
                therapeutic = song.get('therapeutic_value', {})
                text += f"   🎯 Effetto: {therapeutic.get('primary_effect', 'N/A')}\n"
                text += f"   💭 Impatto: {therapeutic.get('emotional_impact', 'N/A')}\n"
                text += f"   🌟 Setting: {therapeutic.get('listening_setting', 'N/A')}\n"
                text += f"   ⏰ Momento: {therapeutic.get('best_moment', 'N/A')}\n"
                text += f"   📝 Motivazione: {song.get('reason', 'N/A')}\n\n"
            
            return text
        except Exception as e:
            return f"Errore nella formattazione delle raccomandazioni: {str(e)}"

def initialize_session():
    """Initialize Streamlit session state"""
    if 'client' not in st.session_state:
        api_key = openaikey
        if not api_key:
            st.error("API key non trovata. Controlla il file .env")
            st.stop()
        st.session_state.client = OpenAI(api_key=api_key)
    
    if 'history' not in st.session_state:
        st.session_state.history = []

def main():
    """Main application function"""
    st.set_page_config(page_title="Volumio Music Assistant", page_icon="🎵", layout="wide")
    st.title("🎵 Volumio Music Assistant")
    st.write("Raccontami come ti senti e ti suggerirò la musica più adatta al tuo stato d'animo.")
    
    initialize_session()
    
    # Audio recording
    st.write("🎤 Registra il tuo messaggio vocale")
    audio_bytes = audio_recorder()
    
    if audio_bytes:
        with st.spinner("Elaborazione in corso..."):
            try:
                # Speech recognition
                processor = AudioProcessor()
                text = processor.process(audio_bytes)
                
                if text:
                    st.info(f"📝 Trascrizione: {text}")
                    
                    # Emotional analysis
                    analyzer = EmotionalAnalyzer(st.session_state.client)
                    analysis = analyzer.analyze(text)
                    
                    if analysis:
                        # Music recommendations
                        recommender = MusicRecommender(st.session_state.client)
                        recommendations = recommender.recommend(analysis)
                        
                        # Format and display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            analysis_text = ResponseFormatter.format_analysis(analysis)
                            st.markdown(analysis_text)
                        
                        with col2:
                            recommendations_text = ResponseFormatter.format_recommendations(recommendations)
                            st.markdown(recommendations_text)
                        
                        # Text to speech
                        audio = AudioRenderer.render(recommendations_text)
                        if audio:
                            st.audio(audio)
                        
                        # Save to history
                        st.session_state.history.append({
                            "text": text,
                            "analysis": analysis_text,
                            "recommendations": recommendations_text
                        })
            
            except Exception as e:
                st.error(f"Si è verificato un errore: {str(e)}")
    
    # Display history
    if st.session_state.history:
        with st.expander("📚 Storico Sessioni", expanded=False):
            for i, session in enumerate(st.session_state.history, 1):
                st.subheader(f"Sessione {i}")
                st.write(f"Hai detto: {session.get('text', 'N/A')}")
                st.write(session.get('analysis', ''))
                st.write(session.get('recommendations', ''))
                st.divider()

if __name__ == "__main__":
    main()