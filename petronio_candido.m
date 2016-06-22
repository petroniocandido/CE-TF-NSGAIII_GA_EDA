% Aplicação aos Problemas DTLZ1 e DTLZ2 com 3 e 5 objetivos

% Entradas:
%  naval -> número de avaliações da função objetivo
%  optP  -> seleção do problema: 1 - DTLZ1 / Outros - DTLZ2
%  nobj  -> número de objetivos (3 ou 5 objetivos)
%  nexec -> número de execuções do algortimo para solução do problema

% Saídas:
%  xBest   -> matriz contendo as váriaveis dos individuos não dominados da execução com melhor IGD
%  yBest   -> matriz contendo a avaliação das funções objetivo para cada individuo de iBest
%  IGDbest -> melhor valor de IGD obtido (relativo a xBest)
%  IGDmed  -> média dos valores de IGD obtidos para as 'nexec' execuções
%  IGDwort -> pior valor de IGD obtido

function [xBest, yBest, IGDbest, IGDmed, IGDworst] = petronio_candido(naval, optP, nobj, nexec)
end

function flag = domina(ObjSol1,ObjSol2)
      
% ObjSol1 -> vetor com valores de avaliação das funções objetos do indivíduo 1
% ObjSol2 -> vetor com valores de avaliação das funções objetos do indivíduo 2

    % Se as soluções são incomparáveis, retorna 0
    flag = 0;
    
    % Se houver relação de dominância, retorna o indice indicando o indivíduo dominante
    
    % a) solução 1 domina a solução 2
    if all(ObjSol2 >= ObjSol1) && any(ObjSol1 < ObjSol2)
       flag = 1;
        
    % b) solução 2 domina a solução 1
    elseif all(ObjSol1 >= ObjSol2) && any(ObjSol2 < ObjSol1)
       flag = 2;
    end
    
end

function xPoP = FastNonDominatedSort(xPoP, nobj, nvar, npop)

    % Indivíduos não dominados recebem rank = 1;
    % Indivíduos da segunda fronteira recebem rank = 2;
    % E assim sucessivamente...
    % Após ordenar fronteiras, calcula a distância de aglomeração para cada frente.
 
    % Non-Dominated Sort: Ordena a População Baseado em Não Dominância

    % inicializa primeira frente (não dominada)  
    frente = 1;
    F(frente).f = [];
    individuo = [];

    % posição para para guardar dados de rank (número da frente de dominância)
    pos = nobj + nvar + 1;      

    % 1) Compara a dominância entre todos os individuos da população, 
    %    dois a dois, e identifica a frente de não dominância.

    % para cada individuo "i" da população...
    for i = 1:npop

        individuo(i).n = 0;     % número de indivíduos que dominam "i" 
        individuo(i).p = [];    % conjunto que guardará todos indivíduos que "i" domina;

        % toma valores das funções objetivo para o indivíduo "i"
        Obj_Xi = xPoP(i,nvar+1:nvar+nobj);

        % para cada individuo "j" da população...
        for j = 1:npop

            % toma valores das funções objetivo para o indivíduo "j"
            Obj_Xj = xPoP(j,nvar+1:nvar+nobj);

            % verifica dominância
            flag = domina(Obj_Xi,Obj_Xj);

            % se "j" domina "i": incrementa o número de indivíduos que o dominam;
            if flag == 2
                individuo(i).n = individuo(i).n + 1;

            % então "i" domina "j": guarda índice do indivíduo "j" dominado por "i"
            elseif flag == 1
                individuo(i).p = [individuo(i).p j];
            end
        end   

        % se solução não for dominada por nenhuma outra... 
        % esta solução pertence a frente não dominada (rank=1)
        if individuo(i).n == 0
            xPoP(i, pos) = 1;                   % guarda rank na população
            F(frente).f = [F(frente).f i];      % salva individuo da frente não dominada
        end
    end

    % 2) Divide as demais soluções pelas frentes de dominância,
    %    conforme memória de dominação entre indivíduos da população

    % encontra as fronteiras seguintes:
    while ~isempty(F(frente).f)

       Qf = [];  % conjunto para guardar individuos da i-ésima fronteira;

       % para cada indivíduo "i" pertencente a última fronteira Fi de não dominância verificada...   
       for i = 1:length(F(frente).f)

           if ~isempty(individuo(F(frente).f(i)).p)

               % para cada um dos "j" indivíduos dominados pelos membros de Fi...
               for j = 1:length(individuo(F(frente).f(i)).p)

                   % decrementa o contador de dominação do indivíduo "j"
                   individuo(individuo(F(frente).f(i)).p(j)).n = individuo(individuo(F(frente).f(i)).p(j)).n - 1;

                   % verifica que nenhum dos individuos nas fronteiras subsequentes dominam "q"
                   if individuo(individuo(F(frente).f(i)).p(j)).n == 0

                        xPoP(individuo(F(frente).f(i)).p(j),pos) = frente + 1;   % guarda rank do indivíduo
                        Qf = [Qf individuo(F(frente).f(i)).p(j)];                % salva indivíduo da frente não dominada atual
                   end                
               end
           end
       end

       % atualiza posição para guardar indivíduos da próxima frente de dominância
       frente =  frente + 1;

       % salva individuos na frente de verificação corrente
       F(frente).f = Qf;

    end
end

%--------------------
% Função para Cálculo da Distância de Multidão
function xPoP = CrowdingDistance(xPoP, nobj, nvar, nsel)

    % Crowding Distance: Calcula distância de multidão 
    % (distância para as soluções vizinhas em cada dimensão do espaço de busca)

    % posição com informação do rank (número da frente de dominância)
    pos = nobj + nvar + 1;    

    % ordena individuos da população conforme nível de dominância
    [~,indice_fr] = sort(xPoP(:,pos));
    xPoP = xPoP(indice_fr,:);
       
    % verifica qual a ultima frente de dominância a entrar diretamente na população de pais
    lastF =  xPoP(nsel,pos);
    
    % verifica qual o pior rank de frente de não dominância na população
    worstF = max(xPoP(:,pos));
    
    % encontra a distância de multidão para cada indivíduo das frentes de dominância selecionadas
    for frente = 1:lastF

        % indice do primeiro individuo da fronteira
        if frente==1
            indice_ini = 1;    
        else
            indice_ini = indice_fim+1;
        end

        % indice do último indivíduo pertencente a fronteira
        if frente~=lastF || lastF < worstF
            indice_fim = find(xPoP(:,pos)>frente,1)-1;    
        else
            indice_fim = length(xPoP);
        end
        
        % número de soluções na frente
        nsolFi = indice_fim-indice_ini+1;

        % separa apenas as avaliações de função objetivo dos individuos na fronteira Fi
        xPop_Fi = xPoP(indice_ini:indice_fim, nvar+1:nvar+nobj);

        % inicializa vetor com valor nulo para as distâncias de multidão
        Di=zeros(1,nsolFi);

        % para cada objetivo "i"...
        for i = 1 : nobj

            % ordena indivíduos da fronteira baseado no valor do objetivo "i"  
            [~, indice_obj] = sort(xPop_Fi(:,i));
            
            % maior valor para o objetivo "i" - último indice
            f_max = xPop_Fi(indice_obj(end),i);

            % menor valor para o objetivo "i" - primeiro indice
            f_min = xPop_Fi(indice_obj(1),i);

            % atribui valor "infinito" para indivíduos na extremidade ótima da fronteira Fi
            Di(1,indice_obj(1)) = Di(1,indice_obj(1)) + Inf;

            % para individuos entre que não estão nas extremidades...
            for j = 2 : nsolFi

                % identifica valores da função objetivos das soluções vizinhas
                if j~=nsolFi
                    proximo   = xPop_Fi(indice_obj(j+1),i);
                else
                    % no extremo máximo, soma o semiperimetro apenas entre o único vizinho
                    proximo   = xPop_Fi(indice_obj(j),i);
                end

                anterior  = xPop_Fi(indice_obj(j-1),i);

                % calcula semi-perimetro normalizado
                Di(1,indice_obj(j)) = Di(1,indice_obj(j))+(proximo - anterior)/(f_max - f_min);
            end      
        end

        % guarda distâncias calculadas por individuo
        xPoP(indice_ini:indice_fim,pos+1)=Di';

        % ordena individuos da frente conforme distância de multidão
        [~,indice_fr] = sort(Di,'descend');
        xPoP(indice_ini:indice_fim,:) = xPoP(indice_fr+(indice_ini-1),:);
    end
    
    % seleciona apenas o valor 'nsel' de soluções
    xPoP = xPoP(1:nsel,:);
end

%--------------------
% Funcao para Cálculo da Distância Geracional Invertida (IGD)
function IGD = CalculaIGD(pareto, solucao)
    
    % entradas:
    % pareto - Soluções da Fronteira de Pareto
    % solucao - Soluções não dominadas obtidas pelo algoritmo em teste

    npareto = length(pareto);   % núm. de soluções da fronteira de Pareto
    nsol = length(solucao);     % núm. de soluções obtidas pelo algoritmo desenvolvido
    
    dmin = zeros(1,npareto);    % guarda menores distâncias (di) entre a fronteira pareto e as soluções não dominadas
    d = zeros(nsol,npareto);    % dist. euclidiana entre cada ponto da fronteira pareto e cada solução não dominada
    
    % calcula distância euclidiana ponto a ponto
    for i=1:npareto
        for j=1:nsol            
            d(i,j) = norm(pareto(i,:)-solucao(j,:),2);
        end
        
        % guarda menor distância
        dmin(i) = min(d(i,:));
    end
    
    % realiza a média das menores distâncias
    IGD = mean(dmin);
end
 
