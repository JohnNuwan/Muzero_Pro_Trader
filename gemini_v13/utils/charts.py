import math

def generate_ascii_chart(data, height=10, width=60):
    """
    Génère un graphique ASCII simple à partir d'une liste de valeurs.
    """
    if not data or len(data) < 2:
        return ""

    # Limiter la largeur
    if len(data) > width:
        data = data[-width:]
        
    min_val = min(data)
    max_val = max(data)
    range_val = max_val - min_val
    
    if range_val == 0:
        range_val = 1
        
    step = range_val / height
    
    chart = []
    
    # Construction du graphique ligne par ligne (de haut en bas)
    for h in range(height, -1, -1):
        line = ""
        threshold = min_val + (h * step)
        
        # Axe Y (Prix)
        label = f"{threshold:.5f}"
        line += f"{label:>10} | "
        
        # Points
        for i, val in enumerate(data):
            # On détermine la position relative de ce point
            # Si la valeur est proche de ce niveau 'h', on met un caractère
            # C'est une approx simple
            normalized_val = (val - min_val) / range_val * height
            
            if math.floor(normalized_val) == h:
                # Déterminer la direction pour le caractère
                if i > 0:
                    prev = data[i-1]
                    if val > prev:
                        line += "↗" # Hausse
                    elif val < prev:
                        line += "↘" # Baisse
                    else:
                        line += "─" # Stable
                else:
                    line += "•"
            else:
                line += " "
                
        chart.append(line)
        
    # Axe X (Bas)
    chart.append(" " * 12 + "└" + "─" * len(data))
    
    return "\n".join(chart)
