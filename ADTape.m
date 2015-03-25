classdef ADTape < handle
    properties
        x
        tape = {};
    end
    
    methods
        function ad = ADTape(x)
            ad.x = ADNode(ad, x);
        end
   
        function add(ad, t)
            ad.tape{end+1} = t;
        end
        
        function backprop(ad)
            for k = length(ad.tape):-1:1
                ad.tape{k}.backprop;
                ad.tape(k) = [];
            end
        end
    end
end