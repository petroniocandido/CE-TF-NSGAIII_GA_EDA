%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	         COMPUTAÇÃO EVOLUCIONÁRIA - TRABALHO FINAL     
%	Programa de Pós Graduação em Engenharia Elétrica - PPGEE
%	Universidade Federal de Minas Gerais - UFMG
%
%	Prof.: João Vasconcelos
%	Aluno: Petrônio Cândido de Lima e Silva
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Aplicação aos Problemas DTLZ1 e DTLZ2 com 3 e 5 objetivos

% Entradas:
%  naval 	= número de avaliações da função objetivo
%  problema	= seleção do problema: 1 - DTLZ1 / Outros - DTLZ2
%  nobj		= número de objetivos (3 ou 5 objetivos)
%  nexec	= número de execuções do algortimo para solução do problema

% Saídas:
%  xBest	= matriz contendo as váriaveis dos individuos não dominados da execução com melhor IGD
%  yBest	= matriz contendo a avaliação das funções objetivo para cada individuo de iBest
%  igd_max	= melhor valor de IGD obtido (relativo a xBest)
%  igd_mean	= média dos valores de IGD obtidos para as 'nexec' execuções
%  igd_min	= pior valor de IGD obtido

% P = [ nvar , nobj , frente , penalidades ] x nbpop

function [xBest, yBest, igd_max, igd_mean, igd_min] = petronio_candido(naval, problema, nobj, nexec)
	NCOL = nvar + nobj + 2;
	
	% CR - CONSTANTE DE CRUZAMENTO
    PC = 1;
    
    % CM - CONSTANTE DE MUTAÇÃO
    PM = 0.5;
    
    % número de individuos da população
	if nobj == 3
		NBPOP = 91;
	else
		NBPOP = 210;
	end
	
	%
	if problema == 1 
		nvar = 5;
	else
		nvar = 10;
	end
	
	ixObjetivos = nvar + nobj;
	
	% número de gerações
	ngen= round(naval/NBPOP);

	for Solucao=1:nexec
		% GERAR POPULAÇÃO INICIAL
		P = geraPopulacaoInicial(NBPOP, nvar, NCOL);
		
		% AVALIAR POPULAÇÃO INICIAL
		P = avaliarPopulacao(P, NBPOP, nvar, nobj, problema);	
		
		% Ordena população por frentes de não-dominância 
		% & calcula distancia de multidão entre individuos da mesma frente
		P = FastNonDominatedSort(P,nobj,nvar,NBPOP);
		P = CrowdingDistance(P,nobj,nvar,NBPOP);
		
		while geracao <= ngen 
			geracao = geracao + 1;
			
			% SELEÇÃO
			% Q = (...)
			
			% CRUZAMENTO
			% Q = (...)
			
			% MUTAÇÃO
			% Q = (...)
			
			% AVALIAR POPULAÇÃO
			Q = avaliarPopulacao(Q, NBPOP, nvar, nobj, problema);
			
			% S = P U Q
			S = [P(:,1:(nvar+nobj));Q];
			
			% Ordena individuos por frentes de não dominância
			S = FastNonDominatedSort(S,nobj,nvar,NBPOP*2);
			
			% Calcula distância de multidão e seleciona 50% dos melhores indivíduos em St
			P = CrowdingDistance(S,nobj,nvar,NBPOP);
			
		end
		
		% verifica o número de soluções não dominadas finais obtidas
		if P(end,ixObjetivos+1) == 1
			nnd = npop;
		else
			nnd = find(P(:,ixObjetivos+1)>1,1)-1;   
		end
			
		solucoes_var = P(1:nnd,1:nvar);            % valores das variáveis de busca para a solução final
		solucoes_obj = P(1:nnd,nvar+1:ixObjetivos);  % variáveis da solução final
				
		% Calcula IGD das Soluções
		if optP == 1 && nobj==3        
		   load('dtlz1_3d.mat');
		   igd(Solucao) = IGD(fronteiraReal, solucoes_obj);

		elseif optP == 1 && nobj==5        
		   load('dtlz1_5d.mat');
		   igd(Solucao) = IGD(fronteiraReal, solucoes_obj);
		   
		elseif optP ~= 1 && nobj==3          
		   load('dtlz2_3d.mat');
		   igd(Solucao) = IGD(fronteiraReal, solucoes_obj);
		   
		else
		   load('dtlz2_5d.mat');
		   igd(Solucao) = IGD(fronteiraReal, solucoes_obj);       
		end   
			
	%     % plota solução final
	%     figure()
	%     hold off
	%     plot3(sol_obj(:,1),sol_obj(:,2),sol_obj(:,3),'or');
	%     hold on
	%     plot3(fronteiraReal(:,1),fronteiraReal(:,2),fronteiraReal(:,3),'*b');
		
		sfinal(Solucao).var = solucoes_var;
		sfinal(Solucao).obj = solucoes_obj;
		
    end
    %  Retorna atributos requisitados:
    
    % a) melhor solução
    [igd_max,id] = min(igd);        
    xBest = sfinal(id).var;        % variáveis    
    yBest = sfinal(id).obj;        % objetivos 
    
    % IGD médio
    igd_mean = mean(igd);
    
    % Pior IGD
    igd_min = max(igd);
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GERA ALEATORIAMENTE A POPULAÇÃO INICIAL
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function P  = geraPopulacaoInicial(nbpop, nvar, ncol)
	P = zeros(nbpop, ncol);
    
    P(:,1:nvar) = rand(nbpop, nvar);
       
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [Pnew] = avaliarPopulacao(Pold, nbpop, nvar, nobj, tipo) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Pnew = avaliarPopulacao(Pold, nbpop, nvar, nobj, tipo) 
	if tipo == 1
        Pnew = avaliarPopulacaoDTLZ1 (Pold,nbpop,nvar,nobj);
    else
        Pnew = avaliarPopulacaoDTLZ2 (Pold,nbpop,nvar,nobj);
    end
end

% Funcao de Avaliação - DTLZ1
function Pnew = avaliarPopulacaoDTLZ1 (Pold,nbpop,nvar,nobj)
    
    k=5;
    f = zeros(1,nobj);
    
    for ind=1:nbpop

        x = Pold(ind,:);
    
        s = 0;

        for i = nobj:nvar
            s = s + (x(i)-0.5)^2 - cos(20*pi*(x(i)-0.5));
        end

        g = 100*(k+s);

        f(1) = 0.5 * prod(x(1:nobj-1)) * (1+g);
        
        for i = 2:nobj-1
            f(i) = 0.5 * prod(x(1:nobj-i)) * (1-x(nobj-i+1)) * (1+g);
        end

        f(nobj) = 0.5 * (1-x(1)) * (1+g);
        
        Pnew(ind,nvar+1:nvar+nobj) = f;
        
    end
end

% Funcao de Avaliação - DTLZ2
function Pnew = avaliarPopulacaoDTLZ2 (Pold,nbpop,nvar,nobj)
    
    k=10;
    f = zeros(1,nobj);

    for ind=1:nbpop
        
        x = Pold(ind,:);

        s = 0;
        
        for i = nobj:nvar
            s = s + (x(i)-0.5)^2;
        end
        
        g = s;

        cosx = cos(x*pi/2);
        sinx = sin(x*pi/2);

        f(1) =  (1+g) * prod(cosx(1:nobj-1));
        
        for i = 2:nobj-1
            f(i) = (1+g) * prod(cosx(1:nobj-i)) * sinx(nobj-i+1);
        end
        
        f(nobj) = (1+g) * sinx(1);
        Pnew(ind,nvar+1:nvar+nobj) = f;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNÇÃO DE DOMINÂNCIA
% flag = 0 - as soluções são incomparáveis
% flag = 1 - u domina v
% flag = 2 - v domina u
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function flag = domina(u,v)

    % As soluções são incomparáveis
    flag = 0;
    
    % u domina v
    if all(u >= v) && any(u < v)
       flag = 1;
        
    % v domina u
    elseif all(u >= v) && any(v < u)
       flag = 2;
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FAST NON DOMINATED SORTING
% Ordena a População Baseado em Não Dominância
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function P = FastNonDominatedSort(P, nobj, nvar, nbpop)

    % Indivíduos não dominados recebem rank = 1;
    % Indivíduos da segunda fronteira recebem rank = 2;
    % E assim sucessivamente...
    % Após ordenar fronteiras, calcula a distância de aglomeração para cada frente.
 
    % inicializa primeira frente (não dominada)  
    frente = 1;
    F(frente).f = [];
    individuo = [];

	% índices na matriz de população P
    ixRanking = nobj + nvar + 1;     %frente de dominância
    ixObj = nvar + nobj;		%final da faixa de objetivos

    % 1) Compara a dominância entre todos os individuos da população, 
    %    dois a dois, e identifica a frente de não dominância.

    % para cada individuo "i" da população...
    for i = 1:nbpop

        individuo(i).n = 0;     % número de indivíduos que dominam "i" 
        individuo(i).p = [];    % conjunto que guardará todos indivíduos que "i" domina;

        % toma valores das funções objetivo para o indivíduo "i"
        xi = P(i,nvar+1:ixObj);

        % para cada individuo "j" da população...
        for j = 1:nbpop

            % toma valores das funções objetivo para o indivíduo "j"
            xj = P(j,nvar+1:ixObj);

            % verifica dominância
            flag = domina(xi,xj);

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
            P(i, ixRanking) = 1;                   % guarda rank na população
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

                        P(individuo(F(frente).f(i)).p(j),ixRanking) = frente + 1;   % guarda rank do indivíduo
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CROWDING DISTANCE
% Função para Cálculo da Distância de Multidão: distância para as soluções vizinhas em cada dimensão do espaço de busca
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function P = CrowdingDistance(P, nobj, nvar, nsel)

    % posição com informação do rank (número da frente de dominância)
    ixRanking = nvar + nobj + 1;    

    % ordena individuos da população conforme nível de dominância
    [~,indice_fr] = sort(P(:,ixRanking));
    P = P(indice_fr,:);
       
    % verifica qual a ultima frente de dominância a entrar diretamente na população de pais
    lastF =  P(nsel,ixRanking);
    
    % verifica qual o pior rank de frente de não dominância na população
    worstF = max(P(:,ixRanking));
    
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
            indice_fim = find(P(:,ixRanking)>frente,1)-1;    
        else
            indice_fim = length(P);
        end
        
        % número de soluções na frente
        nsolFi = indice_fim-indice_ini+1;

        % separa apenas as avaliações de função objetivo dos individuos na fronteira Fi
        P_Fi = P(indice_ini:indice_fim, nvar+1:nvar+nobj);

        % inicializa vetor com valor nulo para as distâncias de multidão
        Di=zeros(1,nsolFi);

        % para cada objetivo "i"...
        for i = 1 : nobj

            % ordena indivíduos da fronteira baseado no valor do objetivo "i"  
            [~, indice_obj] = sort(P_Fi(:,i));
            
            % maior valor para o objetivo "i" - último indice
            f_max = P_Fi(indice_obj(end),i);

            % menor valor para o objetivo "i" - primeiro indice
            f_min = P_Fi(indice_obj(1),i);

            % atribui valor "infinito" para indivíduos na extremidade ótima da fronteira Fi
            Di(1,indice_obj(1)) = Di(1,indice_obj(1)) + Inf;

            % para individuos entre que não estão nas extremidades...
            for j = 2 : nsolFi

                % identifica valores da função objetivos das soluções vizinhas
                if j~=nsolFi
                    proximo   = P_Fi(indice_obj(j+1),i);
                else
                    % no extremo máximo, soma o semiperimetro apenas entre o único vizinho
                    proximo   = P_Fi(indice_obj(j),i);
                end

                anterior  = P_Fi(indice_obj(j-1),i);

                % calcula semi-perimetro normalizado
                Di(1,indice_obj(j)) = Di(1,indice_obj(j))+(proximo - anterior)/(f_max - f_min);
            end      
        end

        % guarda distâncias calculadas por individuo
        P(indice_ini:indice_fim,ixRanking+1)=Di';

        % ordena individuos da frente conforme distância de multidão
        [~,indice_fr] = sort(Di,'descend');
        P(indice_ini:indice_fim,:) = P(indice_fr+(indice_ini-1),:);
    end
    
    % seleciona apenas o valor 'nsel' de soluções
    P = P(1:nsel,:);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Funcao para Cálculo da Distância Geracional Invertida (IGD)
% pareto - Soluções da Fronteira de Pareto
% solucao - Soluções não dominadas obtidas pelo algoritmo em teste
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function IGD = CalculaIGD(pareto, solucao)
    
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
 
