##  RESCUER AGENT
### @Author: Tacla (UTFPR)
### Demo of use of VictimSim
### Not a complete version of DFS; it comes back prematuraly
### to the base when it enters into a dead end position


from vs.abstract_agent import AbstAgent
from vs.constants import VS
from map import Map
import random
import heapq


## Classe que define o Agente Rescuer com um plano fixo
class Rescuer(AbstAgent):
    def __init__(self, env, config_file):
        """ 
        @param env: a reference to an instance of the environment class
        @param config_file: the absolute path to the agent's config file"""

        super().__init__(env, config_file)

        # Specific initialization for the rescuer
        self.map = Map()            # only SOC_1 has all maps (it is the master)
        self.victims = {}           # list of found victims
        self.plan = []              # a list of planned actions
        self.plan_x = 0             # the x position of the rescuer during the planning phase
        self.plan_y = 0             # the y position of the rescuer during the planning phase
        self.plan_visited = set()   # positions already planned to be visited 
        self.plan_rtime = self.TLIM # the remaing time during the planning phase
        self.plan_walk_time = 0.0   # previewed time to walk during rescue
        self.x = 0                  # the current x position of the rescuer when executing the plan
        self.y = 0                  # the current y position of the rescuer when executing the plan
        self.explorers_remaining = {"EXP_1", "EXP_2", "EXP_3"} # control explorers
        self.rescuers = []          # list of all rescuers
                
        # Starts in IDLE state.
        # It changes to ACTIVE when the map arrives
        self.set_state(VS.IDLE)

    def set_rescuers(self, rescuers_lst):
        """ each rescuer has the reference to the others"""
        self.rescuers = rescuers_lst
        
    def do_rescue(self, map, clusters):
        """ O agente socorrista executa a estratégia de salvamento tendo
            o mapa e os clusters que foram atribuídos a ele.
        """
        # It changes to ACTIVE when the map arrives
        self.set_state(VS.ACTIVE)

        self.map = map
        self.plan = []

        current_pos = (0, 0)

        # Agora 'clusters' é uma lista de IDs (ex: [14, 5, 22])
        for victim_id in clusters:
            
            # 1. BUSCA A COORDENADA DA VÍTIMA USANDO O ID
            if victim_id not in self.victims:
                continue  # Se por segurança o ID não estiver mapeado, pula
                
            # Lembra que a estrutura é: self.victims[id] = ((x, y), sinais_vitais)
            victim_pos, _ = self.victims[victim_id]

            # 2. CALCULA O CAMINHO COM A* USANDO AS COORDENADAS
            path = self.a_star(current_pos, victim_pos)

            if path is None or len(path) < 2:
                continue

            # 3. MONTA OS PASSOS (dx, dy) PARA O PLANO
            for i in range(len(path) - 1):
                p1 = path[i]
                p2 = path[i + 1]

                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]

                # Se for o último passo para chegar NESTA vítima, 
                # pode ser útil salvar como True para o agente saber que chegou e deve socorrer
                is_last_step = (i == len(path) - 2)
                self.plan.append((dx, dy, is_last_step))

            # Atualiza a posição atual para ser a posição desta vítima que acabou de ser "alcançada"
            current_pos = victim_pos

        # Retorno para a Base (0,0) após processar todas as vítimas do cluster
        if current_pos != (0, 0):
            path_back = self.a_star(current_pos, (0, 0))

            if path_back and len(path_back) >= 2:
                for i in range(len(path_back) - 1):
                    p1 = path_back[i]
                    p2 = path_back[i + 1]
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    # Retornando para a base, não é parada de vítima, então deixamos False
                    self.plan.append((dx, dy, False))
        
        
    def merge_maps(self, exp_name, map, victims):
        """ The explorer named exp_name sends the map containing the walls and
        victims' location. The rescuer becomes ACTIVE. From now,
        the deliberate method is called by the environment"""

        # Merge received map directly into self.map
        # Merge all visited coordinates from this explorer into self.map
        for coord, cell_data in map.map_data.items():  
            # Since each explorer contributes visited cells,
            # simply add coordinates not yet present
            if not self.map.in_map(coord):
                difficulty, victim_seq, actions_res = cell_data
                self.map.add(coord, difficulty, victim_seq, actions_res)
    
        print(f"{self.NAME}: Map received from explorer {exp_name}")

        # Merge found victims
        #print()
        #print(f"{self.NAME} Found victs by {exp_name}: {victims}")
        self.victims.update(victims)
        #print(f"{self.NAME} Updated victs: {self.victims}")
        
        # Mark this explorer as received
        self.explorers_remaining.discard(exp_name)

        if self.explorers_remaining:
            print(f"{self.NAME}: Waiting for remaining explorers... {self.explorers_remaining}")
            return
    
        
        # print the merged map
        #self.map.draw()
        
        #### AQUI VAMOS FAZER A CLASSIFICAÇÃO E REGRESSÃO 

        import pandas as pd

        # 1. Preparar a lista de dados para o DataFrame
        dados_vitimas = []

        for seq, (coord, vs) in self.victims.items():
            x, y = coord
            
            # Monta a linha base com ID e coordenadas
            linha = {
                "ID" : seq,
                "pos_x" : x,
                "pos_y" : y
            }
            
            NOMES_SINAIS=[
                "idade",  # 0  → ex: 74
                "fc",     # 1  → ex: 50
                "fr",     # 2  → ex: 115
                "pas",    # 3  → ex: 20
                "spo2",   # 4  → ex: 104
                "s6",     # 5  → ex: 95  (sinal desconhecido)
                "temp",   # 6  → ex: 38.07 (float, temperatura real)
                "pr",     # 7  → ex: 1
                "sg",     # 8  → ex: 2
                "fx",     # 9  → ex: 1
                "queim",  # 10 → ex: 0
                "gcs",    # 11 → ex: 13
                "avpu",   # 12 → ex: 1
            ]

            # Adiciona os sinais vitais dinamicamente ou por índice
            # Se 'vs' for uma lista, podemos desmembrar os sinais vitais:
            if isinstance(vs, list):
                for i, sinal in enumerate(vs):
                    if i < len(NOMES_SINAIS):
                        linha[NOMES_SINAIS[i]] = sinal
            elif isinstance(vs, dict):
                for chave, valor in vs.items():
                    linha[chave] = valor
            else:
                linha["Sinais_Vitais_Raw"] = vs
                
            dados_vitimas.append(linha)

        import joblib
        from sklearn.tree import DecisionTreeClassifier

        # 2. Criar o DataFrame e ordenar por ID da Vítima
        df = pd.DataFrame(dados_vitimas)
        df = df.sort_values(by="ID").reset_index(drop=True)

        features = joblib.load('features_class_reg.pkl')

        df_class_reg = df[features]

        # Carregar os modelos uma única vez
        modelo_class = joblib.load('modelo_rn_atualizado_laura.joblib')
        modelo_regressao = joblib.load('modelo_rn_regressao_laura.joblib')
        scaler_regressao = joblib.load('scaler_regressao.joblib')

        # --- PRIMEIRA PREDIÇÃO (Classificação - tri) ---

        # Realizar a predição e adicionar diretamente no df original
        df['tri'] = modelo_class.predict(df_class_reg)
        print(f"{self.NAME}: Predições gravadas na coluna 'tri'!")

        # --- SEGUNDA PREDIÇÃO (Regressão - sobr) ---

        X_regressao_normalizado = pd.DataFrame(
        scaler_regressao.transform(df_class_reg), 
        columns=df_class_reg.columns
        )

        # Realizar a predição de sobrevivência
        df['sobr'] = modelo_regressao.predict(X_regressao_normalizado)
        df['sobr'] = df['sobr'].clip(lower=0.0, upper=1.0)
        print(f"{self.NAME}: Predições gravadas na coluna 'sobr'!")

        # --- SALVAR O ARQUIVO FINAL ---
        # Salva uma única vez com todas as colunas novas incluídas (tri e sobr)
        df.to_csv("data_teste.csv", index=False, sep=";", encoding="utf-8")
        print("Arquivo data_teste.csv gerado e atualizado com sucesso!")

        ##################
        ### CLUSTERING ###
        ##################
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        # O agente socorrista mestre faz o clustering
        features = ["tri", "sobr", "pos_x", "pos_y"]
        X_cluster = df[features]

        # normalização
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X_cluster)

        # aplica DBSCAB
        modelo_dbscan = DBSCAN(eps=0.5, min_samples=3, metric='euclidean')
        clusters_labels = modelo_dbscan.fit_predict(X_norm)

        # 1. Corrigir o cálculo do ruído (usando clusters_labels) e contar os clusters
        n_clusters = len(set(clusters_labels)) - (1 if -1 in clusters_labels else 0)
        print(f"Número total de clusters encontrados: {n_clusters}")

        # 2. Contar a quantidade exata em cada cluster e no ruído
        valores, contagens = np.unique(clusters_labels, return_counts=True)

        print("\n--- Distribuição das Vítimas por Cluster ---")
        for val, qtd in zip(valores, contagens):
            if val == -1:
                print(f"Pontos de Ruído (Vítimas isoladas): {qtd}")
            else:
                print(f"Cluster {val}: {qtd} vítimas")

        df['cluster'] = clusters_labels

        #printa tabela do relatório
        tabela = df.groupby("cluster")["sobr"].agg(
            quantidade="count",
            minimo="min",
            maximo="max",
            media="mean",
            desvio_padrao="std"
        )

        print(tabela)

         # ─────────────────────────────────────────────
        # ALGORITMO GENÉTICO
        # ─────────────────────────────────────────────
        POP_SIZE    = 150
        N_GEN       = 500
        P_CROSS     = 0.85
        P_MUT       = 0.15
        TOURNAMENT  = 3
        ELITISM     = 2
        BASE_X, BASE_Y = 0, 0   # ajuste para a posição real da base
        COST_READ      = 2.0
        COST_FIRST_AID = 1.0
        COST_LINE      = 1.0
        COST_DIAG      = 1.5

        def euclidean_cost(x1, y1, x2, y2):
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            diag_steps = min(dx, dy)
            line_steps = max(dx, dy) - diag_steps
            return diag_steps * COST_DIAG + line_steps * COST_LINE

        def urgency_weight(tri, sobr):
            return 1.0 + (tri / 3.0) * 0.5 + (1.0 - sobr) * 0.5

        def compute_fitness(order, victim_info):
            if not order:
                return 0.0
            total = 0.0
            v0 = victim_info[order[0]]
            total += euclidean_cost(BASE_X, BASE_Y, v0["x"], v0["y"])
            total += COST_READ + COST_FIRST_AID
            for i in range(1, len(order)):
                va = victim_info[order[i-1]]
                vb = victim_info[order[i]]
                travel = euclidean_cost(va["x"], va["y"], vb["x"], vb["y"])
                position_factor = 1.0 + (i / len(order)) * urgency_weight(vb["tri"], vb["sobr"]) * 0.3
                total += travel * position_factor
                total += COST_READ + COST_FIRST_AID
            vn = victim_info[order[-1]]
            total += euclidean_cost(vn["x"], vn["y"], BASE_X, BASE_Y)
            return total

        def ox1_crossover(p1, p2):
            n = len(p1)
            a, b = sorted(random.sample(range(n), 2))
            child = [None] * n
            child[a:b+1] = p1[a:b+1]
            segment = set(p1[a:b+1])
            fill = [g for g in p2 if g not in segment]
            idx = 0
            for i in range(n):
                if child[i] is None:
                    child[i] = fill[idx]
                    idx += 1
            return child

        def mutate(individual):
            ind = individual[:]
            i, j = sorted(random.sample(range(len(ind)), 2))
            if random.random() < 0.5:
                ind[i], ind[j] = ind[j], ind[i]    # swap
            else:
                ind[i:j+1] = ind[i:j+1][::-1]      # 2-opt
            return ind

        def tournament_select(population, fitnesses):
            candidates = random.sample(range(len(population)), TOURNAMENT)
            best = min(candidates, key=lambda i: fitnesses[i])
            return population[best][:]

        def greedy_individual(victim_ids, victim_info):
            remaining = sorted(
                victim_ids,
                key=lambda vid: -(victim_info[vid]["tri"] / 3.0 + (1 - victim_info[vid]["sobr"]))
            )
            route = [remaining.pop(0)]
            while remaining:
                last = victim_info[route[-1]]
                best = min(
                    remaining,
                    key=lambda vid: euclidean_cost(
                        last["x"], last["y"],
                        victim_info[vid]["x"], victim_info[vid]["y"]
                    ) / (1.0 + urgency_weight(victim_info[vid]["tri"], victim_info[vid]["sobr"]) * 0.2)
                )
                route.append(best)
                remaining.remove(best)
            return route

        def run_ga(victim_ids, victim_info, agent_id):
            if not victim_ids:
                return [], 0.0

            print(f"\n{'='*50}")
            print(f"  AG - Agente {agent_id} | {len(victim_ids)} vítimas")
            print(f"{'='*50}")

            # População inicial
            population = [greedy_individual(victim_ids, victim_info)]
            population.append(sorted(
                victim_ids,
                key=lambda vid: -(victim_info[vid]["tri"] / 3.0 + (1 - victim_info[vid]["sobr"]))
            ))
            while len(population) < POP_SIZE:
                ind = list(victim_ids)
                random.shuffle(ind)
                population.append(ind)

            fitnesses = [compute_fitness(ind, victim_info) for ind in population]
            best_ind  = population[min(range(len(population)), key=lambda i: fitnesses[i])][:]
            best_fit  = min(fitnesses)

            print(f"  Gen   0 | best={best_fit:.2f} | avg={sum(fitnesses)/len(fitnesses):.2f}")

            for gen in range(1, N_GEN + 1):
                new_pop = []

                # Elitismo
                elite_idx = sorted(range(len(population)), key=lambda i: fitnesses[i])[:ELITISM]
                for idx in elite_idx:
                    new_pop.append(population[idx][:])

                # Nova geração
                while len(new_pop) < POP_SIZE:
                    p1 = tournament_select(population, fitnesses)
                    p2 = tournament_select(population, fitnesses)
                    child = ox1_crossover(p1, p2) if random.random() < P_CROSS else p1[:]
                    if random.random() < P_MUT:
                        child = mutate(child)
                    new_pop.append(child)

                population = new_pop
                fitnesses  = [compute_fitness(ind, victim_info) for ind in population]

                cur_best = min(range(len(population)), key=lambda i: fitnesses[i])
                if fitnesses[cur_best] < best_fit:
                    best_fit = fitnesses[cur_best]
                    best_ind = population[cur_best][:]

                if gen % 100 == 0:
                    print(f"  Gen {gen:3d} | best={best_fit:.2f} | avg={sum(fitnesses)/len(fitnesses):.2f}")

            print(f"  Fitness final: {best_fit:.2f}")
            return best_ind, best_fit

        # Montar dicionário no formato esperado pelo AG
        victim_info = {}
        for _, row in df.iterrows():
            victim_info[int(row["ID"])] = {
                "x":    int(row["pos_x"]),
                "y":    int(row["pos_y"]),
                "tri":  int(row["tri"]),
                "sobr": float(row["sobr"]),
            }

        # ─────────────────────────────────────────────
        # ENVIAR CLUSTERS ORDENADOS PARA OS RESCUERS
        # ─────────────────────────────────────────────
        clusters_ordenados = []

        for i in range(n_clusters):
            victim_ids = df[df['cluster'] == i]["ID"].tolist()
            best_order, best_fit = run_ga(victim_ids, victim_info, agent_id=i + 1)
            clusters_ordenados.append(best_order)
            print(f"Cluster {i+1} - ordem final: {best_order}")

        # Vítimas de ruído (cluster == -1): atribui ao rescuer com menos vítimas
        ruido = df[df['cluster'] == -1]["ID"].tolist()
        if ruido:
            print(f"\nAtenção: {len(ruido)} vítimas isoladas redistribuídas ao menor cluster.")
            if clusters_ordenados:
                menor = min(range(len(clusters_ordenados)), key=lambda i: len(clusters_ordenados[i]))
                clusters_ordenados[menor].extend(ruido)

        meu_mapeamento = {
            0: [13, 8, 6, 1, 4, 11],     # Socorrista 1 (índice 0) recebe o Cluster 1
            1: [12, 3, 7, 0, 5],     # Socorrista 2 (índice 1) recebe o Cluster 0
            2: [10, 2, 14, -1, 9]      # Socorrista 3 (índice 2) recebe o Cluster 2
        }

        # Se você quiser dar MAIS de um cluster para o mesmo socorrista, basta colocar na lista:
        # 0: [1, 4],  # Socorrista 1 recebe os clusters 1 e 4 sequencialmente
        
        # Inicializa a lista de 3 rotas vazias
        rotas_finais = [[], [], []]

        # Distribui os clusters exatamente como você escolheu no dicionário
        for idx_socorrista, lista_de_clusters in meu_mapeamento.items():
            for num_cluster in lista_de_clusters:
                # Segurança: verifica se o cluster escolhido realmente existe na lista de gerados
                if num_cluster < len(clusters_ordenados):
                    cluster_ids = clusters_ordenados[num_cluster]
                    rotas_finais[idx_socorrista].extend(cluster_ids)

        # Envia a rota final acumulada para cada um dos 3 socorristas
        for i in range(3):
            self.rescuers[i].victims = self.victims.copy()
            self.rescuers[i].map = self.map
            self.rescuers[i].do_rescue(self.map, rotas_finais[i])
            
        
    def deliberate(self) -> bool:
        """ This is the choice of the next action. The simulator calls this
        method at each reasonning cycle if the agent is ACTIVE.
        Must be implemented in every agent
        @return True: there's one or more actions to do
        @return False: there's no more action to do """

        # No more actions to do
        if self.plan == []:  # empty list, no more actions to do
           print(f"{self.NAME} has finished the plan")
           return False

        # Takes the first action of the plan (walk action) and removes it from the plan
        dx, dy, there_is_vict = self.plan.pop(0)
        #print(f"{self.NAME} pop dx: {dx} dy: {dy} vict: {there_is_vict}")

        # Walk - just one step per deliberation
        walked = self.walk(dx, dy)

        # Rescue the victim at the current position
        if walked == VS.EXECUTED:
            self.x += dx
            self.y += dy
            #print(f"{self.NAME} Walk ok - Rescuer at position ({self.x}, {self.y})")
            # check if there is a victim at the current position
            if there_is_vict:
                rescued = self.first_aid() # True when rescued
                if rescued:
                    print(f"{self.NAME} Victim rescued at ({self.x}, {self.y})")
                else:
                    print(f"{self.NAME} Plan fail - victim not found at ({self.x}, {self.x})")
        else:
            print(f"{self.NAME} Plan fail - walk error - agent at ({self.x}, {self.x})")
            
        #input(f"{self.NAME} remaining time: {self.get_rtime()} Tecle enter")

        return True

    def heuristic(self, node, goal):
        #distancia diagonal entre a posição atual e a posição final (obejtivo) 
        dx = abs(node[0] - goal[0])
        dy = abs(node[1] - goal[1])
        return self.COST_LINE * max(dx, dy) + (self.COST_DIAG - self.COST_LINE) * min(dx, dy)

    def a_star(self, start, goal):
        #A* para encontrar o menor caminho

        open_set = [] # fila de prioridade: (f_cost, g_cost, (x, y), path)
        heapq.heappush(open_set, (0, 0, start, [start]))
        
        g_costs = {start: 0} # dicionário para manter o menor g_cost já encontrado para cada nó
        
        # todas as direções de movimento possíveis:
        directions = [
            (0, -1), (1, -1), (1, 0), (1, 1),
            (0, 1), (-1, 1), (-1, 0), (-1, -1)
        ]
        
        while open_set:
            f, g, current, path = heapq.heappop(open_set)
            
            # se chegou ao destino
            if current == goal:
                return path
                
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # consulta o mapa para saber a dificuldade:
                cell_data = self.map.get(neighbor)
                
                # se é parede, ignora; caso none só para tratamento de possíveis erros
                if cell_data is None or cell_data[0] == VS.OBST_WALL:
                    continue
                    
                difficulty = cell_data[0]
                
                # calcula o custo de cada passo dado:
                base_cost = self.COST_LINE if dx == 0 or dy == 0 else self.COST_DIAG
                step_cost = base_cost * difficulty
                new_g = g + step_cost
                
                # se encontramos um caminho mais barato para o vizinho
                if neighbor not in g_costs or new_g < g_costs[neighbor]:
                    g_costs[neighbor] = new_g
                    h = self.heuristic(neighbor, goal)
                    f = new_g + h
                    heapq.heappush(open_set, (f, new_g, neighbor, path + [neighbor]))
                    
        return None # retorna None se não houver caminho possível    
