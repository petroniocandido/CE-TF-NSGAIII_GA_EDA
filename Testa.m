format short;
clear all;
close all;
clc;

nexec = 8;

% Definir problema [optP] e numero de objetivos [nobj]
% optP: 1 - DTLZ1 / Outros - DTLZ2
% nobj: 3 ou 5 objetivos

for optP = [1]
    for nobj = [5]
        
        % define número de cálculos (baseado no artigo Deb & Jain)
        if (optP == 1)
            if (nobj == 3)
                naval = 91 * 400;
            else
                naval = 210 * 600;
            end
        else
            if (nobj == 3)
                naval = 91 * 250;
            else
                naval = 210 * 350;
            end
        end
        
        % chamada da função
        t = cputime();
        
        [xbest, ybest, IGDbest, IGDmean, IGDworst] = petronio_candido(naval, optP, nobj, nexec)
        
        t = cputime() - t;
        
        % imprime resultados
        fprintf('Problema %d (%d objetivos) IGD - mínimo: %2.2e\tmédio: %2.2e\tmáximo: %2.2e\t\t (%2.2f segundos/exec)\n',optP,nobj,IGDbest,IGDmean,IGDworst,t/nexec);
        
        % carrega fronteira pareto real do problema
        prob = strcat('dtlz',num2str(optP),'_',num2str(nobj),'D.mat');
        load(prob);
        
        % plota gráfico dos resultados
        if (nobj == 3)
            figure();
            hold on;
            plot3(fronteiraReal(:,1),fronteiraReal(:,2),fronteiraReal(:,3),'.b','linewidth',2);
            plot3(ybest(:,1),ybest(:,2),ybest(:,3),'or');
            grid on;
            view(60,30);
            if (optP == 1)
                axis(0.5 * [0 1 0 1 0 1]);
            else
                axis([0 1 0 1 0 1]);
            end
            drawnow();
            legend('Fronteira Pareto', 'Soluções Não Dominadas')
            xlabel('f1');
            ylabel('f2');            
            zlabel('f3');            
        else
            figure();
            parallelcoords(ybest);
            xlabel('Objetivo');
            ylabel('Valor');            
        end
    end
end