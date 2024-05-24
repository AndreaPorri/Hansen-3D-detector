import open3d as o3d
from model_pc import *
from icp_align import preprocess_point_cloud, draw_registration_result

# Funzione per una pre-registrazione globale che non dipenda dalla trasformazione rigida iniziale
def global_registration(source_norm:o3d.geometry.PointCloud, target_norm:o3d.geometry.PointCloud, source_fpfh:o3d.geometry.PointCloud, target_fpfh:o3d.geometry.PointCloud, threshold:float, max_iteration:int):
    """
    Esegue una registrazione globale utilizzando le caratteristiche FPFH, così da ottenere un allineamento non
    dipendente da una trasformazione rigida.

    Args:
        source: PointCloud modello source.
        target: PointCloud del target.
        source_fpfh: Caratteristiche FPFH della nuvola di punti source.
        target_fpfh: Caratteristiche FPFH della nuvola di punti target.
        threshold: Valore utilizzato per calcolare soglia di distanza per la registrazione RANSAC

    Return:
        Restituisce il risultato della registrazione globale RANSAC.
    """
    # Imposta la soglia di distanza per la registrazione RANSAC
    distance_threshold = threshold * 1.5

    
    print(":: RANSAC registration on point clouds.")
    print("   We use a free distance threshold of %.3f." % distance_threshold)
    
    # Esegue la registrazione globale RANSAC basata sul matching di caratteristiche
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_norm, target_norm, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration, 0.999))
    
    return result

# Funzione per singola registrazione globale tra coppie di modello-target point clouds
def single_alignment(source:o3d.geometry.PointCloud, source_norm:o3d.geometry.PointCloud, source_fpfh:o3d.geometry.PointCloud, threshold:float, input_folder:str, output_folder:str, count:int, max_iteration:int) -> int:
    """
    Allinea una singola nuvola di punti di origine (modello) con la prima point cloud presente nella cartella di input_folder,
    successivamente applica la trasformazione, trovata con la registrazione globale RANSAC, al modello. Successivamente, se richiesto, 
    salva la point cloud risultante del nuovo modello nella output_folder, cancella la target point cloud dalla input_foldere ed infine
    restituisce il contatore aggionrnato.

    Args:
        -   source (o3d.geometry.PointCloud): Point cloud del modello da allineare.
        -   source_norm (o3d.geometry.PointCloud): Point cloud del modello dove sono state calcolate anche le normali.
        -   source_fpfh (o3d.geometry.PointCloud): Point cloud del modello dove sono state calcolate anche le features FPFH.
        -   threshold (float): Soglia necessaria per la global_registration.
        -   input_folder (str): Percorso della cartella di input che conterrà le point cloud target.
        -   output_folder (str): Percorso della cartella di output dove verranno salvati i nuovi modelli allineati.
        -   count (int): Contatore per il numero di allineamenti eseguiti.
        -   max_iteration (int): Numero massimo di iterazioni per il registro globale RANSAC.

    Ritorna:
        Contatore aggiornato.
    """
     
    # Ottiene il nome del primo file contenuto nella cartella di input
    file_name = os.listdir(input_folder)[0]
    
    # Ottiene il path assoluto del file
    file_path = os.path.join(input_folder, file_name)
    
    # Se il file esiste
    if os.path.isfile(file_path):
        
        # Carica la point cloud target
        target = o3d.io.read_point_cloud(file_path)
        
        # Preprocessa la point cloud restituendo la point cloud con le features FPFH e le normali
        target_norm, target_fpfh = preprocess_point_cloud(target, threshold)

        print(f'Global registration {count}:')
        
        # Esegue la registrazione globale RANSAC tra la coppia source-target point clouds
        result_ransac = global_registration(source_norm, target_norm, source_fpfh, target_fpfh, threshold, max_iteration)
            
        # Stampa il grafico della registrazione RANSAC
        draw_registration_result(source, target, result_ransac.transformation, color_width, color_height)
        print(f'\t\t-> Done the {count}° global registration')

        # Applica la trasformazione (ottenuta con Ransac) alla source point cloud
        registered_source = source.transform(result_ransac.transformation)

        # Unisce le source and target point clouds allineate
        merged_point_cloud = registered_source + target

        # Chiede all'utente di confermare il merge se viene reputato soddisfacente l'allineamento
        user_input = input('\nWrite \'ok\' if you want to save the new object model:\n')

        # Definisce il path di salvataggio della merged point cloud
        path_ran = "{}/model_{}.ply".format(output_folder,count)
        # Verifica se l'utente ha inserito 'ok'
        if user_input.lower() == 'ok':
            # Esegue il salvataggio
            try:
                o3d.io.write_point_cloud(path_ran, merged_point_cloud, format='ply')
                print(f"\nThe new model {count} has been successfully saved.\n\n")
            except Exception as e:
                print(f"An error occurred while saving the file:", e)
        else:
            print(f"\nThe merge number {count} has not been confirmed. The new model will not be saved.\n\n")
        
        # Rimuove il file della point cloud target dalla cartella di input
        os.remove(file_path)

        # Aggiorna count
        count += 1

        return count
        

    
##########################################################################################################################################

#                                                        ENTRY POINT

##########################################################################################################################################

if __name__ == "__main__":
    ### Estrazione parametri di configurazione ###
    # Path del file di configurazione Yaml
    pathConfiguratorYaml = 'C:/Users/andre/OneDrive/Desktop/MAGISTRALE/Progetto_ADIP/configuration.yaml'
    # Assegna i valori della tupla alle rispettive variabili
    color_width, color_height, depth_width, depth_height, _, _, _, _, _, _, _, _, _, model_path, _, threshold, _, max_iteration, input_folder, folder_model_path = load_hyperparams(pathConfiguratorYaml)

    ### Inizializzazione delle variabili ###
    count = 0
    flag = True


    ### Controllo e pulizia input folder ###
    # Lista file presenti nella cartella di input
    files = os.listdir(input_folder)
    # Verifica se la cartella di input contenga solo file contenenti point cloud ed elimina gli altri
    if files:
        for file in files:
            if not file.endswith('.ply'):
                os.remove(os.path.join(input_folder, file))
        if not os.listdir(input_folder):
            raise ValueError("The input folder does not contain any point cloud")
    else:
        raise ValueError("Input folder is empty")

    
    ### Selezione modalità di inizio allineamento ###
    # Verifica se il percorso esiste come file
    if not os.path.exists(folder_model_path):
        # Crea la directory per i salvataggi di merged point cloud
        os.makedirs(folder_model_path)
    else:
        response = input('\n Do you want to start with the last saved model? (Y/N)\n')
        if response.lower() == 'n':
            for file in os.listdir(folder_model_path):
                os.remove(os.path.join(folder_model_path, file))
            print('\t-> The alignment will start from the base model\n\n')
        else:
            if os.listdir(folder_model_path):
                print('\t-> The alignment will start from the most updated model\n\n')
            else:
                print('\t-> The alignment will start from the base model (advanced models absent)\n\n')

    
    ### CREAZIONE NUOVO MODELLO CONNETTORE HANSEN ###
    while flag:
        
        # Allineamento dal modello base
        if not os.listdir(folder_model_path):
            # Carica il modello source
            source = o3d.io.read_point_cloud(model_path)
            # Trova le point cloud source con le normali e le features FPFH
            source_norm, source_fpfh = preprocess_point_cloud(source, threshold)
            # Allinea il modello base con la prima point cloud presente nella cartella di input e restituisce il nuovo modello se approvato
            count = single_alignment(source, source_norm, source_fpfh, threshold, input_folder, folder_model_path, count, max_iteration)
        
        # Allineamento dal modello avanzato
        else:
            # Carica il modello source più aggiornato
            file_name = os.listdir(folder_model_path)[-1]
            file_path = os.path.join(folder_model_path, file_name)
            source = o3d.io.read_point_cloud(file_path)
            # Trova le point cloud source con le normali e le features FPFH
            source_norm, source_fpfh = preprocess_point_cloud(source, threshold)
            # Allinea il modello pre-allineato con la prima point cloud presente nella cartella di input e restituisce il nuovo modello se approvato
            count = single_alignment(source, source_norm, source_fpfh, threshold, input_folder, folder_model_path, count, max_iteration)

        # Stopping Criteria update
        if not os.listdir(input_folder):
            flag = False
    

    