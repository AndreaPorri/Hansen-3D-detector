import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import os
import copy
from model_pc import *

# Funzione di visualizzazione PointClouds
def draw_registration_result(source:o3d.geometry.PointCloud, target:o3d.geometry.PointCloud, transformations, color_width:int, color_height:int, change_color_pc:str = 'no') -> None:  
    """
    Visualizza il risultato della registrazione di due PointCloud.

    Argomenti:
    - source: Oggetto PointCloud che rappresenta la prima nuvola di punti.
    - target: Oggetto PointCloud che rappresenta la seconda nuvola di punti.
    - transformations: Trasformazioni da applicare alla nuvola di punti source per allinearla con target.
    - color_width: Larghezza della finestra di visualizzazione.
    - color_height: Altezza della finestra di visualizzazione.
    - colorized: Specifica se colorare le nuvole di punti. Valori accettabili: 'si' o 'no'. Default: 'no'.

    Return:
    Nessun valore di ritorno. Mostra solo il grafico 
    """

    ### Preprocess per visualizzazione ###
    # Deep copy delle point cloud
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    
    # Se viene scelto di modificare la colorazione originale delle pointcloud
    if change_color_pc == 'si':
        # Modifica colori della deep copy
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])

    # Applica la trasformazione scelta
    source_temp.transform(transformations)
    
    
    ### Creare un visualizzatore ###
    # Visualizzatore
    visualizer = o3d.visualization.Visualizer()
    # Imposta le dimensioni della finestra
    visualizer.create_window(width=color_width, height=color_height)

    # Disegna le geometrie
    visualizer.add_geometry(source_temp)
    visualizer.add_geometry(target_temp)

    # Calcola il centroide delle due nuvole di punti
    source_center = source_temp.get_center()
    target_center = target_temp.get_center()

    # Imposta il punto in cui la telecamera è orientata
    lookat_point = [(source_center[0] + target_center[0]) / 2, 
                    (source_center[1] + target_center[1]) / 2, 
                    (source_center[2] + target_center[2]) / 2] # La telecamera sarà orientata verso il punto medio tra source_center e target_center.
    ctr = visualizer.get_view_control()
    ctr.set_lookat(lookat_point)

    # Calcola la direzione frontale e verso l'alto della camera
    front_direction = [(target_center[0] - source_center[0]), 
                       (target_center[1] - source_center[1]), 
                       (target_center[2] - source_center[2])]
    up_direction = [0, 0, 1]  # Puoi regolare questo in base alle tue esigenze

    # Normalizza i vettori direzionali
    front_direction = np.array(front_direction) / np.linalg.norm(front_direction)
    up_direction = np.array(up_direction) / np.linalg.norm(up_direction)

    # Imposta la direzione frontale e verso l'alto
    ctr.set_front(front_direction)
    ctr.set_up(up_direction)

    # Visualizza le geometrie e aggiorna i setting del visualizer
    visualizer.run()

# Funzione di preprocess delle PointClouds per poter effettuare allineamenti/registration
def preprocess_point_cloud(pcd:o3d.geometry.PointCloud, threshold:float) -> tuple:
    """
    Preelabora una nuvola di punti stimando le normali e calcolando le features FPFH.

    Args:
        pcd: PointCloud in input.
        threshold: Valore utilizzato per ricavare il raggio usato per la stima delle normali ed il calcolo delle feature.

    Return:
        Tupla contenente la PointCloud con normali stimate e un'altra con le feature FPFH calcolate.
    """
    # Effettua una deep copy della PointCloud per evitare la modifica diretta
    pcd_normal = copy.deepcopy(pcd)

    # Rimozione outlier
    pcd_normal, _ = pcd_normal.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=1.5)
    
    # Imposta un valore, pari alla threshold, per il raggio utilizzato per la stima delle normali
    radius_normal = threshold
    print(":: Estimation of normals with search radius %.3f." % radius_normal)
    pcd_normal.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn = 100))
    
    # Calcola le caratteristiche FPFH con un raggio di ricerca
    radius_feature = threshold * 2.5
    print(":: Calculation of FPFH characteristics with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_normal,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return pcd_normal, pcd_fpfh

# Funzione per una pre-registrazione globale che non dipenda dalla trasformazione rigida iniziale
def global_registration(source:o3d.geometry.PointCloud, target:o3d.geometry.PointCloud, source_fpfh:o3d.geometry.PointCloud, target_fpfh:o3d.geometry.PointCloud, threshold:float):
    """
    Esegue una registrazione globale utilizzando le caratteristiche FPFH, così da ottenere un allineamento iniziale
    non dipendente da una trasformazione rigida.

    Args:
        source: PointCloud modello source.
        target: PointCloud del target.
        source_fpfh: Caratteristiche FPFH della nuvola di punti source.
        target_fpfh: Caratteristiche FPFH della nuvola di punti target.
        threshold: Valore utilizzato per calcolare soglia di distanza per la registrazione RANSAC

    Return:
        Restituisce il risultato della registrazione globale.
    """
    # Imposta la soglia di distanza per la registrazione RANSAC
    distance_threshold = threshold * 1.5
    print(":: RANSAC registration on point clouds.")
    print("   We use a free distance threshold of %.3f." % distance_threshold)
    
    # Esegue la registrazione globale RANSAC basata sul matching di caratteristiche
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    
    return result

# Funzione per la Local Registration che utilizza una global registration come inizializzazione per un refinmento locale
def local_registration(icp_selection:str, source:o3d.geometry.PointCloud, target:o3d.geometry.PointCloud, source_norm:o3d.geometry.PointCloud, target_norm:o3d.geometry.PointCloud, threshold:float, result_ransac, max_iteration:int, color_width:int, color_height:int, change_color_pc) -> None:
    '''
    Funzione per eseguire l'allineamento locale tra due PointCloud utilizzando, a scelta, diverse varianti di ICP. Questo avviene a seguito
    di una registrazione globale che serve a fornire un allineamento iniziale, non dipendente da una trasformazione rigida, utilie al corretto
    funzionamento dell'ICP locale.
    
    Args:
        icp_selection (str): Tipo di metodo di ICP da utilizzare: 'point_to_point', 'point_to_plane', 'colored_icp'
        source (o3d.geometry.PointCloud): Point cloud source senza normali
        target (o3d.geometry.PointCloud): Point cloud target senza normali
        source_norm (o3d.geometry.PointCloud): Point cloud source con normali
        target_norm (o3d.geometry.PointCloud): Nuvo cloud target con normali
        threshold (float): Soglia utilizzata per il calcolo della distanza per l'allineamento
        result_ransac: Risultato del RANSAC (global registration)
        max_iteration (int): Numero massimo di iterazioni per l'ICP
        color_width (int): Larghezza dell'immagine per la visualizzazione
        color_height (int): Altezza dell'immagine per la visualizzazione
        change_color_pc (str): Opzione per settare il colore artificiale delle PointCloud
    '''
    ### ICP point to point ###
    if icp_selection == 'point_to_point':
        # Imposta la soglia di distanza
        distance_threshold = threshold * 0.4
        print("Application of ICP Point to Point")
        
        # Applica l'allineamento point to point
        reg_p2p = o3d.pipelines.registration.registration_icp(
                    source, target, distance_threshold, result_ransac.transformation,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))

        # Stampa il risultato dell'allineamento
        print(f'Transformation assessment:\n{reg_p2p}')
        print("The transformation applied is:")
        print(reg_p2p.transformation)
        
        # Visualizza il risultato dell'allineamento
        draw_registration_result(source, target, reg_p2p.transformation, color_width, color_height, change_color_pc)

    
    ### ICP point to plane ###
    if icp_selection == 'point_to_plane':
        # Imposta la soglia di distanza
        distance_threshold = threshold * 0.4
        print("Application of ICP Point to Plane")
        print(type(result_ransac.transformation))
        # Applica l'allineamento
        reg_p2l = o3d.pipelines.registration.registration_icp(
                    source_norm, target_norm, distance_threshold, result_ransac.transformation,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))

        # Stampa il risultato dell'allineamento
        print(f'Transformation assessment:\n{reg_p2l}')
        print("The transformation applied is:")
        print(reg_p2l.transformation)
        
        # Visualizza il risultato dell'allineamento
        draw_registration_result(source, target, reg_p2l.transformation, color_width, color_height, change_color_pc)

    
    ### ICP Colored ###
    if icp_selection == 'colored_icp' and change_color_pc == 'no':
        # Imposta la soglia di distanza per l'ICP colorato
        distance_threshold = threshold * 0.4
        print("Application ofcolored ICP")
        
        # Applica l'allineamento ColoredICP utilizzando PointCloud colorate
        reg_result = o3d.pipelines.registration.registration_colored_icp(
            source_norm, target_norm, distance_threshold, result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))
        
        # Stampa il risultato dell'allineamento
        print(f'Transformation assessment:\n{reg_result}')
        print("The transformation applied is:")
        print(reg_result.transformation)

        # Visualizza il risultato dell'allineamento
        draw_registration_result(source, target, reg_result.transformation, color_width, color_height, change_color_pc)




##########################################################################################################################################

#                                                        ENTRY POINT

##########################################################################################################################################

if __name__ == "__main__":
    
    
    ### Estrazione parametri di configurazione ###
    # Path del file di configurazione Yaml
    pathConfiguratorYaml = 'C:/Users/andre/OneDrive/Desktop/MAGISTRALE/Progetto_ADIP/configuration.yaml'
    # Assegna i valori della tupla alle rispettive variabili
    color_width, color_height, depth_width, depth_height, _, weights_path, _, _, _, _, max_distance_meters, icp_selection, path_env_pc, model_path, check_alignments, threshold, change_color_pc, max_iteration, _, _ = load_hyperparams(pathConfiguratorYaml)

    # Crea la directory per i salvataggi di point cloud dell'enviroment
    create_folder_if_not_exists(path_env_pc) 

    ### Setting Intel Realsense Camera ###
    # Inizializza la fotocamera RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, 30)
    profile = pipeline.start(config)

    # Ottiene il sensore di profondità
    depth_sensor = profile.get_device().first_depth_sensor()
    
    # Imposta la distanza minima a 5 centimetri -> ATTENZIONE: il range massimo del sensore è comunque [0.25,9] metri.
    depth_sensor.set_option(rs.option.min_distance, 50)

    
    ### Inizializzazioni ### 
    # Segmentation Model
    # Inizializza il modello YOLOv8 con i pesi del path specificato
    model = YOLO(weights_path)

    # Filtro profondità massima
    # Ottiene la scala della depth
    depth_scale = depth_sensor.get_depth_scale()
    # Converte la distanza massima consentita, espressa in metri, nella scala del depth_sensor
    max_distance = max_distance_meters / depth_scale

    # Visualizzazione delle immagini (no allineamento pointcloud)
    # Crea finestre di visualizzazione con dimensioni iniziali adatte al frame
    cv2.namedWindow('2D Object Segmentation', cv2.WINDOW_NORMAL) # Grafico che mostra la segmentation nel flusso di RGB frames 
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

            # Crea un oggetto di allineamento al frame RGB
            align_to = rs.stream.color
            align = rs.align(align_to)

            # Acquisisce i frame allineati
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
                         
            # Evita errori dovuti ad un acquisizione errata dei frame da parte della camera
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

            ### Grafico Real Time ###
            # Converte l'immagine di profondità in un'immagine a colori con tre canali
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # Sovrappone i due frame depth e RGB
            overlay = cv2.addWeighted(color_image_for_detection, 0.5, depth_colormap, 0.5, 0)
            # Mostra la depth image e color image combinate ed a colori
            cv2.imshow('Overlapped Images', overlay)        

                
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

                
                # Se viene premuto il tasto 's' esegue l'icp del modello sul frame selezionato
                if cv2.waitKey(1) & 0xFF == ord('s') and detections[0]:

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
                    # Estrae la ROI dall'immagine RGB
                    colors = color_image[y:y2+1, x:x2+1]
                    # Estrae i colori dei pixel della ROI
                    colors_resized = colors.reshape(-1, 3)
                    # Salva la point cloud a colori
                    save_colored_ply('{}/data_{}_color.ply'.format(path_env_pc, count_frame), world_points, colors_resized)

                    # Controlla che la PointCloud esista e sia salvata
                    if not os.path.exists('{}/data_{}_color.ply'.format(path_env_pc, count_frame)):
                        raise ValueError('Check that it saves PointClouds correctly')
                    
                    
                    
                    ### ALLINEAMENTO MODELLO TRAMITE ICP ###
                    ### Carica le point cloud e definisce dei parametri di trasformazione ### 
                    source = o3d.io.read_point_cloud(model_path) # Legge PointCloud del modello
                    target = o3d.io.read_point_cloud('{}/data_{}_color.ply'.format(path_env_pc, count_frame)) # Legge point cloud dell'enviroment dove si vuol cercare il modello

                    # Trasformazione/allineamento rigido
                    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])                   
                    
                    
                    ### Visualizzazione trasformazione rigida iniziale ###
                    if check_alignments == 'si':
                        '''
                        In questo caso si utilizza semplicemente una trasformazione rigida iniziale per allineare la source alla target PointCloud.
                        La valutazione dell'allineamento viene eseguita utilizzando un valore di soglia predefinito, ed infine,
                        il risultato dell'allineamento viene stampato a scopo di valutazione.
                        Questa porzione di codice serve a comprendere/visualizzare la trasformazione iniziale da cui partirà
                        la global registration.
                        '''
                        # Applica la trasformazione rigida a source
                        source.transform(trans_init)

                        # Visualizza graficamente la registrazione/allineamento iniziale tra le due nuvole di punti
                        draw_registration_result(source, target, trans_init, color_width, color_height, change_color_pc)
                        
                        # Valuta la qualità dell'allineamento rigido
                        print("Allineamento rigido iniziale:")
                        rigid_align = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
                        print(rigid_align)

                    
                    ### Preprocess delle PointCloud che servirà ad effettuare correttamente le registration ###
                    source_norm, source_fpfh = preprocess_point_cloud(source, threshold)
                    target_norm, target_fpfh = preprocess_point_cloud(target, threshold)
                    
                    
                    ### Gobal Registration ### 
                    '''
                    Esegue la registrazione globale delle PointCloud utilizzando le features FPFH.
                    Questa si basa sul algoritmo RANSAC per l'allineamento iniziale. 
                    Il risultato di questa registrazione fornisce un'importante trasformazione iniziale che può essere utilizzata 
                    come punto di partenza per ulteriori registrazioni locali.
                    Il risultato della registrazione globale viene stampato e visualizzato tramite la funzione
                    draw_registration_result che mostra il risultato della registrazione graficamente.

                    '''
                    result_ransac = global_registration(source_norm, target_norm, source_fpfh, target_fpfh, threshold)
                    
                    # Valutazione della registrazione globale
                    print(f'RANSAC transformation assessment\n{result_ransac}')
                    # Visualizzazione della registrazione globale
                    draw_registration_result(source, target, result_ransac.transformation, color_width, color_height, change_color_pc)


                    
                    ### Local Registration ###
                    '''
                    Dopo la registrazione globale iniziale si procede con un allineamento tramite ICP. Di seguito sono presenti tre possibili opzioni
                    selezionabili separatamente attraverso il file di configurazione, entrambe eseguono una particolare variante del algoritmo 
                    ICP che procede all'allineamento iterativo fintanto che non si raggiunge una convergenza in un certo numero definito di iterazioni.
                    Per una spiegazione dettagliata osservare i seguenti tutorial:

                    Example: https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html
                    Youtube: https://www.youtube.com/watch?v=CzbETzWgFrc&list=PLkmvobsnE0GEZugH1Di2Cr_f32qYkv7aN&index=10
                    '''
                   
                    local_registration(icp_selection, source, target, source_norm, target_norm, threshold, result_ransac, max_iteration, color_width, color_height, change_color_pc)


            # Aggiornamento frame counter
            count_frame += 1 

            # Mostra i risultati grafici
            cv2.imshow('2D Object Segmentation', color_image_for_detection)
            cv2.imshow('Overlapped Images', overlay)

            # Esce se viene premuto 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    finally:
        # Rilascia le risorse
        pipeline.stop()
        cv2.destroyAllWindows()