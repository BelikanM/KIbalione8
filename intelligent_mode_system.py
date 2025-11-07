"""
Système de Modes Intelligents pour Kibali
Détecte automatiquement le contexte et active les bons outils
"""

def detect_conversation_mode(query: str, uploaded_file_data=None) -> dict:
    """
    Détecte le mode de conversation optimal basé sur la requête
    
    Returns:
        dict: {
            'mode': str,  # scientifique, analyse, visualisation, code, conversation
            'tools': list,  # Outils recommandés
            'tone': str,  # Ton de la réponse
            'format': str  # Format de sortie recommandé
        }
    """
    query_lower = query.lower()
    
    # MODE SCIENTIFIQUE 🔬
    scientific_keywords = [
        'résistivité', 'ert', 'géophysique', 'profondeur', 'eau salée', 'saline',
        'conductivité', 'ohm', 'électrique', 'tomographie', 'géologique',
        'stratigraphie', 'aquifère', 'nappe', 'sol', 'roche', 'argile', 'sable'
    ]
    
    # MODE ANALYSE DE DONNÉES 📊
    analysis_keywords = [
        'analyse', 'statistique', 'moyenne', 'médiane', 'écart-type', 'variance',
        'distribution', 'corrélation', 'tendance', 'anomalie', 'pattern',
        'cluster', 'classification', 'régression', 'prédiction'
    ]
    
    # MODE VISUALISATION 📈
    visualization_keywords = [
        'graphique', 'plot', 'courbe', 'diagramme', 'carte', 'heatmap',
        'histogramme', 'scatter', 'barres', 'camembert', 'visualise',
        'montre', 'affiche', 'dessine', 'trace', 'représente'
    ]
    
    # MODE CODE/TECHNIQUE 💻
    code_keywords = [
        'code', 'python', 'script', 'fonction', 'algorithme', 'programme',
        'automatise', 'génère', 'développe', 'implémente', 'debug',
        'optimise', 'refactor', 'api', 'librairie', 'package'
    ]
    
    # MODE EXTRACTION DE DONNÉES 🔍
    extraction_keywords = [
        'extrait', 'récupère', 'obtiens', 'tableau', 'dataframe', 'csv',
        'excel', 'export', 'sauvegarde', 'structure', 'format', 'liste',
        'valeurs', 'données brutes', 'dump'
    ]
    
    # MODE COMPARAISON ⚖️
    comparison_keywords = [
        'compare', 'différence', 'vs', 'versus', 'contraste', 'similaire',
        'différent', 'meilleur', 'pire', 'supérieur', 'inférieur'
    ]
    
    # MODE INVESTIGATION PROFONDE 🕵️
    investigation_keywords = [
        'pourquoi', 'comment', 'raison', 'cause', 'origine', 'explique',
        'détaille', 'approfondi', 'complet', 'exhaustif', 'fouille',
        'cherche', 'trouve', 'localise', 'identifie'
    ]
    
    # MODE WEB SEARCH 🌐
    web_keywords = [
        'actualité', 'récent', 'nouveau', 'news', 'internet', 'web',
        'recherche', 'google', 'dernière', 'aujourd\'hui', 'cette semaine'
    ]
    
    # MODE CARTOGRAPHIE 🗺️
    map_keywords = [
        'carte', 'itinéraire', 'trajet', 'route', 'navigation', 'localisation',
        'gps', 'coordonnées', 'latitude', 'longitude', 'osm', 'distance'
    ]
    
    # MODE GÉNÉRATION CRÉATIVE 🎨
    creative_keywords = [
        'génère', 'crée', 'image', 'vidéo', 'audio', '3d', 'modèle',
        'illustration', 'design', 'art', 'créatif', 'imagine'
    ]
    
    # Compteur de correspondances
    mode_scores = {
        'scientific': sum(1 for kw in scientific_keywords if kw in query_lower),
        'analysis': sum(1 for kw in analysis_keywords if kw in query_lower),
        'visualization': sum(1 for kw in visualization_keywords if kw in query_lower),
        'code': sum(1 for kw in code_keywords if kw in query_lower),
        'extraction': sum(1 for kw in extraction_keywords if kw in query_lower),
        'comparison': sum(1 for kw in comparison_keywords if kw in query_lower),
        'investigation': sum(1 for kw in investigation_keywords if kw in query_lower),
        'web_search': sum(1 for kw in web_keywords if kw in query_lower),
        'map': sum(1 for kw in map_keywords if kw in query_lower),
        'creative': sum(1 for kw in creative_keywords if kw in query_lower),
    }
    
    # Mode dominant
    dominant_mode = max(mode_scores, key=mode_scores.get)
    max_score = mode_scores[dominant_mode]
    
    # Si aucun mode clair, mode conversation générale
    if max_score == 0:
        dominant_mode = 'conversation'
    
    # Configuration par mode
    mode_configs = {
        'scientific': {
            'mode': 'Scientifique 🔬',
            'tools': [
                'Binary_Analysis',
                'Deep_Binary_Investigation', 
                'ERT_Interpretation',
                'Local_Knowledge_Base',
                'Hybrid_Search',
                'AI_Code_Generator'
            ],
            'tone': 'précis, technique, avec références scientifiques',
            'format': 'Structure avec sections: Contexte, Analyse, Interprétation, Conclusions'
        },
        'analysis': {
            'mode': 'Analyse de Données 📊',
            'tools': [
                'Binary_Analysis',
                'AI_Code_Generator',
                'AI_Plot_Generator',
                'Text_Summarizer',
                'Local_Knowledge_Base'
            ],
            'tone': 'analytique, factuel, avec statistiques',
            'format': 'Tableaux, statistiques clés, insights'
        },
        'visualization': {
            'mode': 'Visualisation 📈',
            'tools': [
                'AI_Plot_Generator',
                'AI_Code_Generator',
                'Image_Analyzer',
                'Binary_Analysis'
            ],
            'tone': 'descriptif, visuel, pédagogique',
            'format': 'Graphiques, légendes, interprétations visuelles'
        },
        'code': {
            'mode': 'Programmation 💻',
            'tools': [
                'AI_Code_Generator',
                'AI_Plot_Generator',
                'Local_Knowledge_Base'
            ],
            'tone': 'technique, précis, avec exemples de code',
            'format': 'Code commenté, explications, alternatives'
        },
        'extraction': {
            'mode': 'Extraction de Données 🔍',
            'tools': [
                'Binary_Analysis',
                'Deep_Binary_Investigation',
                'AI_Code_Generator',
                'Smart_Content_Extractor'
            ],
            'tone': 'structuré, organisé, complet',
            'format': 'Tableaux, listes, formats exportables'
        },
        'comparison': {
            'mode': 'Comparaison ⚖️',
            'tools': [
                'Binary_Analysis',
                'Hybrid_Search',
                'Local_Knowledge_Base',
                'AI_Plot_Generator'
            ],
            'tone': 'objectif, équilibré, avec métriques',
            'format': 'Tableaux comparatifs, graphiques, synthèse'
        },
        'investigation': {
            'mode': 'Investigation Approfondie 🕵️',
            'tools': [
                'Hybrid_Search',
                'Deep_Binary_Investigation',
                'Web_Search_Detailed',
                'Local_Knowledge_Base',
                'Chat_History_Search',
                'Entity_Extractor'
            ],
            'tone': 'exploratoire, détaillé, multi-sources',
            'format': 'Rapport complet avec sources, connexions, insights'
        },
        'web_search': {
            'mode': 'Recherche Web 🌐',
            'tools': [
                'Web_Search',
                'Web_Search_Detailed',
                'Current_News_Search',
                'Smart_Content_Extractor',
                'Hybrid_Search'
            ],
            'tone': 'informatif, à jour, avec sources',
            'format': 'Résumé avec liens, dates, crédibilité'
        },
        'map': {
            'mode': 'Cartographie 🗺️',
            'tools': [
                'OSM_Route_Calculator',
                'Web_Search',
                'Local_Knowledge_Base'
            ],
            'tone': 'pratique, géographique, précis',
            'format': 'Itinéraires, distances, points d\'intérêt'
        },
        'creative': {
            'mode': 'Génération Créative 🎨',
            'tools': [
                'Text_To_Image_Generator',
                'Text_To_Video_Generator',
                'Text_To_Audio_Generator',
                'Text_To_3D_Generator',
                'Image_To_3D_Generator'
            ],
            'tone': 'créatif, descriptif, imaginatif',
            'format': 'Médias générés avec descriptions'
        },
        'conversation': {
            'mode': 'Conversation Générale 💬',
            'tools': [
                'Chat_History_Search',
                'Local_Knowledge_Base',
                'Web_Search',
                'Hybrid_Search'
            ],
            'tone': 'naturel, engageant, adaptatif',
            'format': 'Réponse fluide et contextuelle'
        }
    }
    
    config = mode_configs.get(dominant_mode, mode_configs['conversation'])
    
    # Ajouter le fichier uploadé comme contexte
    if uploaded_file_data:
        config['has_file'] = True
        config['file_path'] = uploaded_file_data.get('physical_path')
        config['file_name'] = uploaded_file_data.get('filename')
    
    # Ajouter scores pour debug
    config['mode_scores'] = mode_scores
    config['query'] = query
    
    return config


def format_mode_prompt(query: str, mode_config: dict) -> str:
    """
    Génère un prompt enrichi basé sur le mode détecté
    """
    mode = mode_config['mode']
    tone = mode_config['tone']
    format_type = mode_config['format']
    tools = mode_config['tools']
    
    prompt = f"""🎯 MODE ACTIVÉ: {mode}

📋 REQUÊTE UTILISATEUR:
{query}

🔧 OUTILS RECOMMANDÉS POUR CE MODE:
{', '.join(tools)}

💡 INSTRUCTIONS SPÉCIFIQUES:
- Ton: {tone}
- Format de sortie: {format_type}
- Utilise les outils les plus pertinents parmi ceux recommandés
- Adapte ta réponse au contexte scientifique/technique si nécessaire
"""
    
    if mode_config.get('has_file'):
        prompt += f"""
📁 FICHIER UPLOADÉ DISPONIBLE:
- Nom: {mode_config['file_name']}
- Chemin: {mode_config['file_path']}
- Utilise ce fichier pour l'analyse si pertinent
"""
    
    prompt += f"""
🎯 TA MISSION:
Réponds de façon {tone}, en utilisant le format: {format_type}

Commence ta réponse maintenant:
"""
    
    return prompt
