import re
import sys
def get_execution_time(log_file):
        
    with open(log_file, "r") as file:
        log_content = file.read()

    # Utilisation d'une expression régulière pour extraire le temps après "Execution ended after"
    match = re.search(r'Execution ended after (\S+)', log_content)
    if match:
        time_after_execution = match.group(1)
        print(f"Temps après l'exécution : {time_after_execution}")
    else:
        print("Aucune correspondance trouvée.")

get_execution_time(sys.argv[1])
