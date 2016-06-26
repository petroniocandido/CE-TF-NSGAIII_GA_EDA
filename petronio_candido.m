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

% P = [ nvar , nobj , frente , distânncia de multidão ] x nbpop

function [xBest, yBest, igd_max, igd_mean, igd_min] = petronio_candido(naval, problema, nobj, nexec)
	format short;
	
	if problema == 1 && nobj==3        
	   load('dtlz1_3d.mat');
	elseif problema == 1 && nobj==5        
	   load('dtlz1_5d.mat');
	elseif problema ~= 1 && nobj==3          
	   load('dtlz2_3d.mat');
	else
	   load('dtlz2_5d.mat');       
	end   
	
	% número de individuos da população
	if nobj == 3
		NBPOP = 91;
	else
		NBPOP = 210;
	end
		
	%
	if problema == 1 
		k = 5;
	else
		k = 10;
	end
	
	nvar = nobj + k - 1;
	
	NCOL = nvar + nobj + 2;
	
	% Ìndices da matriz P
	
	ixVariaveis = 1 : nvar;
	
	ixObjetivos = nvar +1 : nvar + nobj;
	
	ixRanking = nvar + nobj + 1;
	
	% número de gerações
	ngen= round(naval/NBPOP);

	for Solucao=1:nexec
		% GERAR POPULAÇÃO INICIAL
		P = rand(NBPOP*2,nvar); 
		
		% AVALIAR POPULAÇÃO INICIAL
		P = avaliarPopulacao(P, NBPOP*2, nvar, nobj, problema);
		
		P = [P ; P];
		
		P = NSGA2(P, NBPOP, nvar, nobj);
		
		geracao = 0;
		
		while geracao <= ngen 
			geracao = geracao + 1;
			
			%Q = EDA(P, NBPOP, nvar);
			Q = GA(P, NBPOP, nvar, nobj)
			%Q = HIBRIDO(P, NBPOP, nvar, nobj);
						
			S = [P(:,1:nvar);Q(:,1:nvar)];
			
			% AVALIAR POPULAÇÃO
			S = avaliarPopulacao(S, NBPOP*2, nvar, nobj, problema);
			
			P = NSGA2(S, NBPOP, nvar, nobj);
			
			if mod(geracao,100) == 0
				P
			end
			
		end
		
		% verifica o número de soluções não dominadas finais obtidas
		if P(end,ixRanking) == 1
			nnd = NBPOP;
		else
			nnd = find(P(:,ixRanking)>1,1)-1
		end
			
		solucoes_var = P(1:nnd,ixVariaveis);  % valores das variáveis de busca para a solução final
		solucoes_obj = P(1:nnd,ixObjetivos);  % variáveis da solução final
				
		% Calcula IGD das Soluções
		if problema == 1 && nobj==3        
		   igd(Solucao) = IGD(fronteiraReal, solucoes_obj);

		elseif problema == 1 && nobj==5        
		   igd(Solucao) = IGD(fronteiraReal, solucoes_obj);
		   
		elseif problema ~= 1 && nobj==3          
		   igd(Solucao) = IGD(fronteiraReal, solucoes_obj);
		   
		else
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
		
		Solucao
		
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

function P = avaliarPopulacao(P, nbpop, nvar, nobj, tipo) 
	if tipo == 1
        P = avaliarPopulacaoDTLZ1 (P,nbpop,nvar,nobj);
    else
        P = avaliarPopulacaoDTLZ2 (P,nbpop,nvar,nobj);
    end
end

% Funcao de Avaliação - DTLZ1
function P = avaliarPopulacaoDTLZ1 (P,nbpop,nvar,nobj)
    
    k=5;
    f = zeros(1,nobj);
    
    ixObj = nvar+1:nvar+nobj;
    
    for ind=1:nbpop

        x = P(ind,:);
    
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
        
        P(ind,ixObj) = f;
        
    end
end

% Funcao de Avaliação - DTLZ2
function P = avaliarPopulacaoDTLZ2 (P,nbpop,nvar,nobj)
    
    k=10;
    f = zeros(1,nobj);
    
    ixObj = nvar+1:nvar+nobj;

    for ind=1:nbpop
        
        x = P(ind,:);

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
        P(ind,ixObj) = f;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNÇÃO DE DOMINÂNCIA
% flag = 0 - as soluções são incomparáveis
% flag = 1 - u domina v
% flag = 2 - v domina u
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function flag = domina(U,V)
    % Se as soluções são incomparáveis, retorna 0
    flag = 0;
    
    % U > V
    if all(V >= U) && any(U < V)
       flag = 1;
        
    % V > U
    elseif all(U >= V) && any(V < U)
       flag = 2;
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FAST NON DOMINATED SORTING
% Ordena a População Baseado em Não Dominância
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function P = NSGA_FastNonDominatedSort(P, nobj, nvar, nbpop)

    % Indivíduos não dominados recebem rank = 1;
    % Indivíduos da segunda fronteira recebem rank = 2;
    % E assim sucessivamente...
    % Após ordenar fronteiras, calcula a distância de aglomeração para cada frente.
 
    % inicializa primeira frente (não dominada)  
    frente = 1;
    F(frente).f = [];
    individuo = [];

	% índices na matriz de população P
    ixObj = nvar + 1 : nvar + nobj;		%final da faixa de objetivos
    ixRanking = nobj + nvar + 1;     %frente de dominância

    % 1) Compara a dominância entre todos os individuos da população, 
    %    dois a dois, e identifica a frente de não dominância.

    % para cada individuo "i" da população...
    for i = 1:nbpop

        individuo(i).n = 0;     % número de indivíduos que dominam "i" 
        individuo(i).p = [];    % conjunto que guardará todos indivíduos que "i" domina;

        % toma valores das funções objetivo para o indivíduo "i"
        xi = P(i,ixObj);

        % para cada individuo "j" da população...
        for j = 1:nbpop

            % toma valores das funções objetivo para o indivíduo "j"
            xj = P(j,ixObj);

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
% NSGA - II
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function P = NSGA2(P, nbpop, nvar, nobj)
	% Ordena população por frentes de não-dominância 
	P = NSGA_FastNonDominatedSort(P,nobj,nvar,nbpop*2);
	
	% Calcula distância de multidão entre individuos da mesma frente 
	% e seleciona 50% dos melhores indivíduos em P
	P = NSGA2_CrowdingDistance(P,nobj,nvar,nbpop);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CROWDING DISTANCE
% Função para Cálculo da Distância de Multidão: distância para as soluções vizinhas em cada dimensão do espaço de busca
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function P = NSGA2_CrowdingDistance(P, nobj, nvar, nsel)

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

function igd = IGD(pareto, solucao)
    
    % entradas:
    % pareto - Soluções da Fronteira de Pareto
    % solucao - Soluções não dominadas obtidas pelo algoritmo em teste

    % núm. de soluções da fronteira de Pareto
    [npareto,~] = size(pareto);
    
    % núm. de soluções obtidas pelo algoritmo desenvolvido
    [nsol,~] = size(solucao);
    
    dmin = zeros(1,npareto);    % guarda menores distâncias (di) entre a fronteira pareto e as soluções não dominadas
   
    d = zeros(npareto,nsol);    % dist. euclidiana entre cada ponto da fronteira pareto e cada solução não dominada
 
    % calcula distância euclidiana ponto a ponto
    for i=1:npareto
        for j=1:nsol            
            d(i,j) = norm(pareto(i,:)-solucao(j,:),2);
        end
        
        % guarda menor distância
        dmin(i) = min(d(i,:));
    end
    
    % realiza a média das menores distâncias
    igd = mean(dmin);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EDA - Estimative Distribution Algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Q = EDA(P,nbpop,nvar)
	% SELEÇÃO
	%ind = 1:ceil(nbpop*0.2);
	%m1 = max(ind);
	%ind2 = length(ind)+1:(length(ind)+ceil(NBPOP*0.2)); 
	%ind2 = m1+1:2*m1;
	%ind3 = 2*m1+1:nbpop;

	% EDA
	%Q1 = prob1(P(ind,:),0.5,nvar,NBPOP);
	%Q2 = prob1(P(ind2,:),0.5,nvar,NBPOP);
	%[m1, s1] = EDA_estimarParametro_MediaDp_PorVariavel(P(ind,:),nvar)
	%[m2, s2] = EDA_estimarParametro_MediaDp_PorVariavel(P(ind2,:),nvar)
	%[m3, s3] = EDA_estimarParametro_MediaDp_PorVariavel(P(ind3,:),nvar)
	%[ma1, mi1] = estimarParametro_MaxMin(P(ind,:),nvar);
	%[ma2, mi2] = estimarParametro_MaxMin(P(ind2,:),nvar);
	%pq1 = ceil(nbpop*0.6);
	%pq2 = ceil(nbpop*0.3);
	%Q1 = EDA_gerarPopulacao_gaussUVGlobal(m1, s1, pq1,nvar);
	%Q2 = EDA_gerarPopulacao_gaussUVGlobal(m2, s2, pq2,nvar);
	%Q3 = EDA_gerarPopulacao_gaussUVGlobal(m3, s3, nbpop - pq1 - pq2,nvar);
	
	%Q = [Q1;Q2;Q3];
	%pq1 = ceil(nbpop*0.5);
	
	%[m1, s1] = EDA_estimarParametro_MediaDp_PorVariavel(P,nvar)
	%Q = EDA_gerarPopulacao_gaussUV(m1, s1, nbpop, nvar);
	
	[m1, s1] = EDA_estimarParametro_MediaDp_Global(P,nvar)
	Q = EDA_gerarPopulacao_gaussUVGlobal(m1, s1, nbpop, nvar);
	
	Q = max(min(Q,1),0);
end
 
function [MU, SIGMA] = EDA_estimarParametro_MediaDp_Global(P,nvar)
	MU = mean(mean(P(:,1:nvar)));
	SIGMA = max(std(P(:,1:nvar)));
end

% Gaussiana Univariada 
function [MU, SIGMA] = EDA_estimarParametro_MediaDp_PorVariavel(P,nvar)
	MU = mean(P(:,1:nvar));
	SIGMA = std(P(:,1:nvar));
end

function [MU, SIGMA] = EDA_estimarParametro_MuSigma(P,nvar)
	MU = mean(P(:,1:nvar));
	SIGMA = cov(P(:,1:nvar));
end

% Uniforme 
function [MIN,MAX] = EDA_estimarParametro_MaxMin(P,nvar)
	MIN = min(P(:,1:nvar));
	MAX = max(P(:,1:nvar));
end

function P= EDA_gerarPopulacao_gaussUV(mu, sigma,nbpop,nvar)
	for i=1:nbpop
		for k=1:nvar
			P(i,k) = normrnd(mu(k), sigma(k));
		end
	end
end

function P= EDA_gerarPopulacao_lognormUV(mu, sigma,nbpop,nvar)
	for i=1:nbpop
		P(i,1:nvar) = max(min(normrnd(mu, sigma),1),0);
	end
end

function P = EDA_gerarPopulacao_unif(pmin, pmax, nbpop, nvar)
	for k=1:nvar
		prng(k) = pmax(k)-pmin(k);
	end
	for i=1:nbpop
		for k=1:nvar
			P(i,k) = prng(k)*rand()+pmin(k);
		end
	end
end

function P = EDA_gerarPopulacao_unifMDP(mu, sigma,nbpop,nvar)
	pmax = min(mu + sigma,1);
	pmin = max(mu - sigma,0);
	for i=1:nbpop
		P(i,1:nvar) = (pmax-pmin)*rand()+pmin;
	end
end

function P = EDA_gerarPopulacao_gaussMV(mu,sigma,nbpop,nvar)
	P(1:nbpop,1:nvar) = max(min(mvnrnd(mu,sigma,nbpop),1),0);
end

function P = EDA_gerarPopulacao_gaussUVGlobal(mu, sigma,nbpop,nvar)
	for i=1:nbpop
		for k = 1:nvar
			P(i,k) = normrnd(mu, sigma);
		end
	end
end

% Função de Probabilidade baseada em Gausiana Normal
function [m] = EDA_prob1(mx,p,nvar,nBpop)
    desv = std(mx(:,1:nvar));
    med = mean(mx(:,1:nvar));
    liminf = max((med-desv*1),0);
    limsup = min((med+desv*1),1);
    for i=1:nvar
        if i==1
            m_prob = unifrnd(liminf(1,i),limsup(1,i),round(nBpop*p),1);
        else
            m_prob = [m_prob unifrnd(liminf(1,i),limsup(1,i),round(nBpop*p),1)];
        end
    end
    m = m_prob;
end

% Função de Probabilidade baseada em LogNormal
function [m] = EDA_prob2(mx,p,nvar,nBpop)
    desv = std(mx(:,1:nvar));
    med = mean(mx(:,1:nvar));
    for i=1:round(nBpop*p)
        if i==1
            m_prob = min(lognrnd(med,desv),1)*(rand);
        else
            m_prob(i,:) = min(lognrnd(med,desv),1)*(rand);
        end
    end
    m = m_prob;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GA - Genetic Algorithm with real codification
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Q = GA(P,nbpop,nvar,nobj)

	ps = 1;
	pc = 0.9;
	pm = 0.5;
	
	ixRanking = nvar + nobj + 1;
	
	if P(end,ixRanking) == 1
		Q = GA_selecao_torneio(P, nbpop, ixRanking, ps);
	else
		% ELITISMO
		Q1 = P(P(:,ixRanking) == 1, : );
		P2 = P(P(:,ixRanking) > 1, : );
		[l2 ~] = size(P2);

		% SELEÇÃO
		Q2 = GA_selecao_torneio(P2, l2, ixRanking, ps);
		
		Q = [Q1 ; Q2];
	end
	
	
	% CRUZAMENTO
	Q = GA_cruzamento(Q, nbpop, nvar, ixRanking, pc);
	
	Q = max(min(Q,1),0);
	
	% MUTAÇÃO	
	Q = GA_mutacao(Q, nbpop, nvar, pm);
	
	Q = max(min(Q,1),0);
	
	
end


function [I, J] = GA_escolhe2(P,nbpop,ixRanking)
	a = randi(nbpop);	
	b = randi(nbpop);
	
	while P(a,ixRanking) == 0 || P(b,ixRanking) == 0
		a = randi(nbpop);	
		b = randi(nbpop);
	end
			
	if P(a,ixRanking) < P(b,ixRanking)
		I = a;
		J = b;
	else
		I = b;
		J = a;
	end
end

function PNew = GA_selecao_torneio(POld, nbpop, ixRanking, ps)
    num_selecionados = nbpop * ps;
    
    k = 0.9;
    
    for ix = 1:num_selecionados
		[i,j] = GA_escolhe2(POld,nbpop,ixRanking);
		r = rand();
		if r < k
			PNew(ix,:) = POld(i,:);
		else
			PNew(ix,:) = POld(j,:);
		end
    end
end

function PNew = GA_cruzamento(POld, nbpop, nvar, ixRanking, pc)
	PNew = POld;
    
    [r, c] = size(POld);
    
    cruzamentos = r * pc;
    
    alpha_pol = 0.9;	% Coef. de multiplicação linear polarizado
    
    alpha = 0.5;		% Coef. de multiplicação linear 
    
    for ix = 1:cruzamentos
    	    	
		dir = randi([0, 1]);	% Direção do cruzamento
		
		kcross = randi([1, nvar]);	% Ponto de corte
		
		[i,j] = GA_escolhe2(POld,nbpop,ixRanking);
		
		if dir == 0
			tmp1 = (alpha_pol * POld(i, 1:kcross)) + ((1-alpha_pol) * POld(j, 1:kcross));
			f1 =  [tmp1, POld(i, kcross+1:nvar)];
			tmp2 = ((1-alpha) * POld(i, 1:kcross)) + (alpha * POld(j, 1:kcross));
			f2 =  [tmp2, POld(j, kcross+1:nvar)];
		else
			tmp1 = (alpha_pol * POld(i, kcross:nvar)) + ((1-alpha_pol) * POld(j, kcross:nvar));
			f1 =  [POld(i, 1:kcross-1), tmp1];
			tmp2 = ((1-alpha) * POld(i, kcross:nvar)) + (alpha * POld(j, kcross:nvar));
			f2 =  [POld(j, 1:kcross-1), tmp2];
		end
					
		PNew(i,1:nvar) = f1;
		PNew(j,1:nvar) = f2;
		
	end

end

function PNew = GA_mutacao(POld, nbpop, nvar, pm)
	%PNew = zeros(nbpop, ncol);
    PNew = POld;
    [R, s] = size(POld);
    for i = 1:R
    
		r = rand();
		
		if r < pm
		
			if exist('vrange')
				clear vrange;
			end
		
			dir = randi([0, 1]);	% Direção da mutação
			
			kmut = randi(nvar - dir);	% Ponto de mutação
			
			if dir == 0
				for k = 1:kmut
					vrange(k) = max(POld(:,k)) - min(POld(:,k));
					if vrange(k) < 0.2
						vrange(k) = rand();
					end
				end
			else
				for k = kmut:nvar
					vrange(k-(kmut-1)) = max(POld(:,k)) - min(POld(:,k));
					if vrange(k-(kmut-1)) < 0.2
						vrange(k-(kmut-1)) = rand();
					end
				end
			end
    
			beta = -1*rand()+rand();
			
			gamma = 0.5*beta*vrange;
			
			if dir == 0	
				PNew(i,1:kmut) = POld(i,1:kmut) + gamma;
			else
				PNew(i,kmut:nvar) = POld(i,kmut:nvar) + gamma;
			end
		end
    end
    for i = R+1:nbpop
		PNew(i,1:nvar) = rand(1,nvar);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Híbrido entre GA e EDA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Q = HIBRIDO(P,nbpop,nvar,nobj)
	ixRanking = nvar + nobj + 1;
	if P(end,ixRanking) == 1
		Q = EDA(P,nbpop,nvar);
	else
		Q = GA(P,nbpop,nvar,nobj);
	end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NSGA-III
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function P = NSGA3(P, nbpop, nvar, nobj)
	% Ordena população por frentes de não-dominância 
	P = NSGA_FastNonDominatedSort(P,nobj,nvar,nbpop*2);
	
	% Normaliza os objetivos
	
	% Associa com os nichos
	
	[qtd dist d] = NSGA3_associarNichos(P,Z);
	
	% Preservar nichos
	
	P = NSGA3_preservarNichos(P);
end


function objz = NSGA3_normalizarObjetivos(P,nbpop,nvar)
	for i = 1:nvar
		zmin(i) = min( P(:,i) );
		zrng(i) = max( P(:,i) ) - zmin(i);
	end
	for k = 1:nbpop
		for i = 1:nvar
			objz(k,i) = (P(k,i) - zmin(i))/zrng(i)
		end
	end
end

%Operação de Associação
%Parâmetros de entrada: "pops": Funções objetivos "popz": Matriz dos pontos de referências
%Parâmetros de saída: 
%"qtd": Quantidade de soluções associadas ao ponto referencia
%"dist": Solução com menor distância por função objetivo

function [qtd dist d] = NSGA3_associarNichos(P,Z)
    [n m] = size(Z);
    d = [];
    dist = [];
    for i = 1:n
        menor = Inf;
        for j = 1:m
            %Cálculo da distância de projeção entre a solução e o ponto de         %referência
            temp = norm(P(i,:)'-Z(j,:)'*P(i,:)*Z(j,:)'./norm(Z(j,:)')^2);
            if(temp < menor)
               menor = temp;
               indice = j;
            end
        %Matriz de distância da solução para o ponto de referência    
        d(i,j)= temp;  
        end
        %Ponto em que a solução esta associado por frente de dominância
        qtd(i,:) = indice;
        %Solução com menor distância por função objetivo 
        dist(i,:) = menor;
    end
end


function P = NSGA3_preservarNichos(P)
end
