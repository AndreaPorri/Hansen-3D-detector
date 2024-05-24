import os
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import yaml
import copy



# Funzione per la creazione delle directory di salvataggio
def create_folder_if_not_exists(folder_path:str) -> None:
    '''
    Funzione per creare le directory.

    Args:
        folder_path(str): path dove si vuole creare la directory.
    Return: 
        Crea la directory.
    '''
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Funzioni per l'utilizzo dei parametri di configurazione conenuti nel file YAML
def yamlParser(path_yaml_file:str):
    """
    Funzione necessaria per leggere un file YAML e salvarne il contenuto in una variabile. L'input è il path del file .yaml.
    Args:
        - path_yaml_file(str): path del file yaml.
    Return:
        Oggetto parser.
    """
    
    with open(path_yaml_file, "r") as stream: # Apre il file .yaml in modalità lettura
        yaml_parser = yaml.safe_load(stream) # Salva il contenuto in una variabile
    return yaml_parser

def load_hyperparams(pathConfiguratorYaml:str) -> tuple:
    """
    Questa funzione semplifica l'accesso ai valori degli iperparametri del modello nel file YAML. In pratica, la funzione carica e restituisce gli 
    iperparametri caricati in variabili. La funzione ha un solo ingresso: il path del file YAML.
    
    Args:
        - pathConfiguratorYaml(str): path del file yaml.
    Returns:
        Tupla contenente le variabili presenti nel file yaml.
    """
    
    yaml_configurator = yamlParser(pathConfiguratorYaml) # Restituisce uno spazio dei nomi contenente le coppie chiave-valore
                                                         # corrispondenti ai parametri specificati nel file

    # Ogni variabile assume il valore corrispondente al campo specificato nel Namespace:
    color_width = yaml_configurator["color_width"]
    color_height = yaml_configurator["color_height"]
    depth_width = yaml_configurator["depth_width"]
    depth_height = yaml_configurator["depth_height"]
    pc_segframe_path = yaml_configurator["pc_segframe_path"]
    weights_path = yaml_configurator["weights_path"]
    pc_path = yaml_configurator["pc_path"]
    seg_path = yaml_configurator["seg_path"]
    use_visual_preset = yaml_configurator["use_visual_preset"]
    visual_preset_chose = yaml_configurator["visual_preset_chose"]
    max_distance_meters = yaml_configurator["max_distance_meters"]
    icp_selection = yaml_configurator['icp_selection']
    path_env_pc = yaml_configurator['path_env_pc']
    model_path = yaml_configurator['model_path']
    check_alignments = yaml_configurator['check_alignments']
    threshold = yaml_configurator['threshold']
    change_color_pc = yaml_configurator['change_color_pc']
    max_iteration = yaml_configurator['max_iteration']
    input_folder = yaml_configurator['input_folder']
    folder_model_path = yaml_configurator['folder_model_path']
   
    return color_width, color_height, depth_width, depth_height, pc_segframe_path, weights_path, pc_path, seg_path, use_visual_preset, visual_preset_chose, max_distance_meters, icp_selection, path_env_pc, model_path, check_alignments, threshold, change_color_pc, max_iteration, input_folder, folder_model_path

# Funzione che trasforma un array numpy di coord 3D in una point cloud a colori
def save_colored_ply(filename: str, points: np.array, colors: np.array) -> None:
    '''
    Funzione per la scrittura del file .ply che rappresenterà la point cloud a colori, la quale viene ottenuta a partire dall'array numpy di coord 3D fornite in input e un array numpy di colori.

    Args:
        - filename(str): path della directory dove verrà salvato il file.
        - points(array): array dove sono presenti le coordinate xyz di ogni pixel presente all'interno del box di detection.
        - colors(array): array dove sono presenti i valori RGB dei colori di ciascun punto.
    Return:
        File con formato .ply che rappresenta la point cloud dell'oggetto detectato con colori.
    '''
    # Scrive l'header necessario al PLY format
    header = '''ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''.format(len(points))

    # Scrive i data points nel formato corretto
    with open(filename, 'w') as f:
        f.write(header)
        for i, point in enumerate(points):
            color = colors[i]
            if point[0] == 0 and point[1] == 0 and point[2] == 0:
                f.write('{} {} {} {} {} {}\n'.format(point[0], point[1], point[2], 0, 0, 0))
            else:
                f.write('{} {} {} {} {} {}\n'.format(point[0], point[1], point[2], color[2], color[1], color[0]))

# Funzione che trasforma un array numpy di coord 3D in una point cloud
def save_ply(filename: str, points: np.array) -> None:
    '''
    Funzione per la scrittura del file .ply che rappresenterà la point cloud, la quale viene ottenuta a partire dall'array numpy di coord 3D fornite in input.
    
    Args:
        - filename(str): path della directory dove verrà salvato il file.
        - points(array): array dove sono presenti le coordinate xyz di ogni pixel presente all'interno del box di detection.
    Return:
        File con formato .ply che rappresenta la point cloud dell'oggetto detectato.
    '''
    # Scrive l'header necessario al PLY format
    header = '''ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
end_header
'''.format(len(points))

    # Scrive i data points nel formato corretto
    with open(filename, 'w') as f:
        f.write(header)
        for point in points:
            f.write('{} {} {}\n'.format(point[0], point[1], point[2]))



##########################################################################################################################################

#                                                        ENTRY POINT

##########################################################################################################################################

if __name__ == "__main__":
    
    
    ### Estrazione parametri di configurazione ###
    # Path del file di configurazione Yaml
    pathConfiguratorYaml = 'C:/Users/andre/OneDrive/Desktop/MAGISTRALE/Progetto_ADIP/configuration.yaml'
    # Assegna i valori della tupla alle rispettive variabili
    color_width, color_height, depth_width, depth_height, pc_segframe_path, weights_path, pc_path, seg_path, use_visual_preset, visual_preset_chose, max_distance_meters, _, _, _, _, _, _, _, _, _ = load_hyperparams(pathConfiguratorYaml)
    
    # Crea la directory per i salvataggi di point cloud e corrispondenti frame segmentati
    create_folder_if_not_exists(pc_segframe_path)    
    


    ### Setting Intel Realsense Camera ###
    # Inizializza la fotocamera RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, 30)
    profile = pipeline.start(config)

    # Ottiene il sensore di profondità
    depth_sensor = profile.get_device().first_depth_sensor()
    
    '''
    Con questo comando è possibile vedere quali funzioni sono utilizzzabili sfruttando l'oggetto 'depth_sensor':
                            
                                            print(dir(depth_sensor))
    
    Tra quelle disponibili è presente una funzione che permette di vedere tutta una serie di funzionalità presenti in rs.option e
    supportate dal 'device'. Queste permetteranno, con l'utilizzo di depth_sensor.get_option(...) e di depth_sensor.set_option(..), di
    vedere o modificare alcuni setting specifici del depth sensor. La funzione è:
                                            
                                        print(depth_sensor.get_supported_options())

    '''
    
    # Imposta la distanza minima a 5 centimetri -> ATTENZIONE: il range massimo del sensore è comunque [0.25,9] metri.
    depth_sensor.set_option(rs.option.min_distance, 50)
    
    '''
    Per visualizzare i valori appena settati (o quelli standard) basta eseguire questo print:
                                    
                                    min_distance = depth_sensor.get_option(rs.option.min_distance)
                                    print(min_distance)
    
    Mentre per avere una piccola descrizione:
                                    print(depth_sensor.get_option_description(rs.option.min_distance))
    '''

    # Decide se utilizzare o meno dei preset di visualizzazione (Specifiche L515: https://dev.intelrealsense.com/docs/lidar-camera-l515-datasheet   ---> p.8 sez.3.2)
    if use_visual_preset == 'si':    
        # Setta il preset 
        depth_sensor.set_option(rs.option.visual_preset, visual_preset_chose)



    ### Inizializzazioni ### 
    # Segmentation Model
    # Inizializza il modello YOLOv8 con i pesi del path specificato
    model = YOLO(weights_path)

    # Filtro profondità massima
    # Ottiene la scala della depth
    depth_scale = depth_sensor.get_depth_scale()
    # Converte la distanza massima consentita, espressa in metri, nella scala del depth_sensor
    max_distance = max_distance_meters / depth_scale

    # Visualizzazione delle immagini
    cv2.namedWindow('2D Object Segmentation', cv2.WINDOW_NORMAL) # Grafico che mostra la segmentation nel flusso di RGB frames 
    cv2.namedWindow('Depth Frame', cv2.WINDOW_NORMAL) # Grafico che mostra la depth in scala di grigi nel flusso di Depth frames 
    cv2.namedWindow('Object Segmentation Mask', cv2.WINDOW_NORMAL) # Grafico che mostra l'oggetto segmentato con una maschera bianco-nero
    cv2.namedWindow('Overlapped Images', cv2.WINDOW_NORMAL) # Grafico che mostra la sovrapposizione dei depth frames con i corrispondenti RGB frames (Per mostrarne il corretto allineamento)

    # Counter
    # Inizializza il counter del numero di frame
    count_frame = 0

   
                                    ########################################
        
                                                # INIZIO LOOP
        
                                    ########################################

    try:
        while True:
            # Acquisisce il frame dalla fotocamera RealSense (RGB camera e depth frame)
            frames = pipeline.wait_for_frames()

            # Creare un oggetto di allineamento al frame RGB
            align_to = rs.stream.color
            align = rs.align(align_to)

            # Allinea i frame RGB e depth e ne estrae i valori
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
                         
            # Evita errori dovuti ad un acquisizione errata dei frame
            if not color_frame or not depth_frame:
                continue
            
            # Converte i frame in un array numpy
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image_for_detection = copy.deepcopy(color_image) # Crea una copia che verrà utilizzata per la segmentation
            
            
            
            ### Filtra Depth Frame ###
            # Applica il filtro sulla max distance per ignorare le depth che eccedono tale soglia
            depth_image = np.where(depth_image <= max_distance, depth_image, 0)
        
            
            
            ### Object Segmentation ###      
            # Esegue inferenza (segmentazione) con YOLOv8 sul frame RGB per rilevare il connettore
            detections = model(color_image_for_detection)

            # Inizializza la maschera bianco-nero per il tracciamento del oggetto segmentato (inizializzata nera)
            mask = np.zeros((color_height, color_width))

            
            # Loop su ogni detection nella lista dei rilevamenti
            for idx, det in enumerate(detections[0]):
                
                # Estrae dati utili relativi alla singola detection
                x, y, x2, y2 = det.boxes.xyxy[0].cpu() # Coordinate box detection
                seg = det.masks.xy # Punti xy del contorno dell'oggetto segmentato

                # Estrae coordinate box
                x, y, x2, y2 = int(x), int(y), int(x2), int(y2)

                # Converte seg in una lista di array contenenti le coord dei punti
                seg_points = [seg[0].astype(int)]
                
                # Disegna la segmentation e box sul RGB frame
                cv2.rectangle(color_image_for_detection, (x, y), (x2, y2), (255, 0, 0), 2)
                cv2.polylines(color_image_for_detection, seg_points, True, (0, 0, 255), 4)

                # Resize della maschera ottenuta dal frame segmentato
                raw_mask = det.masks.data[0].cpu().numpy()
                resized_mask = cv2.resize(raw_mask, (color_width, color_height)).round()

                # Aggiorna maschera con l'oggetto segmentato
                mask += resized_mask


                ### Object 3D coordinates ###
                # Estrae la ROI dalla depth image
                depth_roi = depth_image[y:y2+1, x:x2+1]

                # Ottiene le informazioni intrinseche della fotocamera di profondità
                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

                # Raccoglie le coordinate (x,y) dei pixel in un array
                depth_points = np.array([[x_current, y_current] for y_current in range(y, y2 + 1) for x_current in range(x, x2 + 1)])  # Costruisci array di coordinate 2D
                # Raccoglie le depth corrispondenti ai pixel in un flattened array
                depth_values = depth_roi.flatten() * depth_scale  # Valori di profondità in metri

                # Calcola le coordinate 3D per ogni punto nella ROI e le salva in una lista di triple
                world_points = []
                for i in range(len(depth_values)):
                    depth_pixel = depth_points[i]
                    depth_value = depth_values[i]
                    point_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, depth_value) # Calcola la coordinata (x,y,z)
                    world_points.append(point_3d)

                # Converte la lista in un array di coordinate 3D
                world_points = np.array(world_points)

                    

                ### Point cloud ###
                # Estrae i colori dei pixel corrispondenti alle coordinate 2D delle ROI dall'immagine RGB
                colors = color_image[y:y2+1, x:x2+1]
                # Riformatta i colori per corrispondere alla dimensione della point cloud
                colors_resized = colors.reshape(-1, 3)
                # Salva la point cloud a colori
                save_colored_ply(pc_path.format(count_frame,idx), world_points, colors_resized)

                ### Salva la detection in pdf ###
                # Salva i frame RGB segmentati 
                cv2.imwrite(seg_path.format(count_frame,idx), color_image_for_detection)
                

            count_frame += 1 # Update frame counter

            ### Grafici Real Time ###
            # Normalizza i valori di depth nell'intervallo 0-255 per la visualizzazione (scala di grigi)
            depth_image_print = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # Converte l'immagine di profondità in un'immagine a colori con tre canali
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # Sovrappone i due frame depth e RGB
            overlay = cv2.addWeighted(color_image, 0.5, depth_colormap, 0.5, 0)        

            # Mostra i risultati grafici
            cv2.imshow('2D Object Segmentation', color_image_for_detection)
            cv2.imshow('Depth Frame', depth_image_print)
            cv2.imshow('Object Segmentation Mask', mask)
            cv2.imshow('Overlapped Images', overlay)

            # Esce se viene premuto 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Rilascia le risorse
        pipeline.stop()
        cv2.destroyAllWindows()