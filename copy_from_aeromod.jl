# cf Table 1 in Wiedensohler 1988 (An approxiamtion of the bipolar charge distribution for particles in the submicron size range)
AiN = [-26.3328 -2.3197 -0.0003 -2.3484 -44.4756;
       35.9044   0.6175 -0.1014  0.6044  79.3772;
       -21.4608  0.6201  0.3073  0.4800 -62.8900;
       7.0867   -0.1105 -0.3372  0.0013  26.4492;
       -1.3088  -0.1260  0.1023 -0.1544 -5.7480;
       0.1051    0.0297 -0.0105 0.0320 0.5049];

s_th_up = 1.0e-6   # [m], upper limit for the validity of AiN coefficient (approximation coef for the less than two charges on a particle)
s_th_down = 1.0e-9 # [m], lower limit for the validity of AiN coefficient


"""

    P_q_charges_knowing_size(q::Int64,s::Array{Cdouble,1};T::Cdouble=293.0,Z_p::Cdouble=Z_p0,Z_m::Cdouble=Z_m0)

    compute the conditional probability P(q|s) (probability of q elementary charges knowing the size s of the particle)

    The probabilities are compute based on the approximation in Wiedensohler 1988 (J. Aerosol Sci., Vol. 19, No. 3, p. 387-389) and should be updated with an adjusted model 

    input:

      - q: number of elementary charges
      - s: particle diameter [m]
      - T: temperature [K] at which the device operates
      - Z_p and Z_m: positive and negative ion electrical mobility

    output

      - P(q|s)
"""
function P_q_charges_knowing_size(q::Int64,s::Array{Cdouble,1};T::Cdouble=293.0,Z_p::Cdouble=Z_p0,Z_m::Cdouble=Z_m0) # s::Union{Cdouble,Array{Cdouble,1}} #WARNING: this is a breaking modification, but not for the end user though
    val = zeros(Cdouble,length(s));

    if (abs(q)<=2)
        if (abs(q)<=1)
            # formula valid for the range [1,1000] nm
            for i in eachindex(s) # 1:length(s)
                if ((s[i]>=1.0e-9) & (s[i]<=s_th_up))
                    tmp = 0.0;
                    for k in 1:6
                        tmp = tmp + AiN[k,q+3]*((log10(1.0e9s[i]))^(k-1))
                    end
                    val[i] = 10.0^tmp
                else
                    # cf Table 2 in Wiedensohler 1988 (An approxiamtion of the bipolar charge distribution for particles in the submicron size range)
                    if (q==0)
                        if (s[i]<=1.0e-9)
                            val[i] = 0.9909;
                        else
                            val[i] = 0.1236;
                        end
                    elseif (q==1)
                        if (s[i]<=1.0e-9)
                            val[i] = 0.0044;
                        else
                            val[i] = 0.1024;
                        end
                    else
                        if (s[i]<=1.0e-9)
                            val[i] = 0.0047;
                        else
                            val[i] = 0.1333;
                        end
                    end
                end
            end
        else
            # formula valid for the range [20,1000] nm
            for i in eachindex(s) # 1:length(s)
                if ((s[i]>=20.0e-9) & (s[i]<=s_th_up))
                    tmp = 0.0;
                    for k in 1:6
                        tmp = tmp + AiN[k,q+3]*((log10(1.0e9s[i]))^(k-1))
                    end
                    val[i] = 10.0^tmp
                else
                    if (q==2)
                        if (s[i]<=20.0e-9)
                            val[i] = 0.0001;
                        else
                            val[i] = 0.0759;
                        end
                    else
                        if (s[i]<=20.0e-9)
                            val[i] = 0.0001;
                        else
                            val[i] = 0.1286;
                        end
                    end
                end
            end
        end
    else
        # Gunn 1956: assuming equal concentration of positive and negative ions
        tmp1 = e_charge./sqrt.(4.0pi^2*eps_0*s*kb*T)
        tmp_exp_num = (q.-2.0pi*eps_0*s*kb*T*log(Z_p/Z_m)/(e_charge^2)).^2;
        tmp_exp_den = 4.0pi*eps_0*s*kb*T/(e_charge^2)
        val = tmp1.*exp.(-tmp_exp_num./tmp_exp_den);
    end
    # # Gunn 1956: assuming equal concentration of positive and negative ions
    # tmp1 = e_charge./sqrt.(4.0pi^2*eps_0*2.5*s*kb*T)
    # tmp_exp_num = (q-2.0pi*eps_0*2.5*s*kb*T*log(Z_p/Z_m)/(e_charge^2)).^2;
    # tmp_exp_den = 4.0pi*eps_0*2.5*s*kb*T/(e_charge^2)
    # val = tmp1.*exp.(-tmp_exp_num./tmp_exp_den);
    val
end

# cut off by inertial impactor
function impactor_eff(s::Union{Cdouble,Array{Cdouble,1}};s50::Cdouble=1.0e-6,delta50::Cdouble=0.1e-6)
    1.0./(1.0.+exp.((s.-s50)/delta50))
end

function impactor(u::Array{Cdouble,1},s::Array{Cdouble,1};s50::Cdouble=1.0e-6,delta50::Cdouble=0.1e-6)
    u.*impactor_eff(s;s50=s50,delta50=delta50)
end

"""
    neutralizer_Kr_85(u_imp::Array{Cdouble,1},s::Array{Cdouble,1};T::Cdouble=293.0,Pr::Cdouble=Pr0,Nq::Int64=6)

    compute the electrical mobilities and charge probability for a neutralizer Kr85 (charge conditioner)

    input:

      - u_imp:    particle size distribution at the inlet of the charge conditioner
      - s:        particle diameter [m] discretization nodes
      - T and Pr: the temperature [K] and pressure [Pa] at which the device operates
      - Nq:       the signed maximal number of chargers acquired by a particle

    output:

      - R_charge: range of number of charge
      - K:        electrical mobilities [m^2 V^{-1} s^{-1}] for all the particle size and each number of charge
      - u_q_k:    particle mobility density [# m^{-3} m^{-2} V s] keeping track of the charging probabilities (row q is the mobility density for q charges varying with particle diameter)
      - Pqs:      probability of q elementary charges for a particle with diameter s
"""
function neutralizer_Kr_85(u_imp::Array{Cdouble,1},s::Array{Cdouble,1};T::Cdouble=293.0,Pr::Cdouble=Pr0,Nq::Int64=6)
    # create the set of mobilities coresponding to up to 6 elementary charges
    K = zeros(Cdouble,abs(Nq),length(s));
    if Nq>0
        R_charge = 1:Nq
    else
        R_charge = -1:-1:Nq
    end

    [K[abs(q),:]=mobility_from_size_and_charge(s,q;T=T,Pr=Pr) for q in R_charge]
    # probability for a particle of having q charges knowing its size
    Pqs = zeros(Cdouble,abs(Nq),length(s));
    [Pqs[abs(q),:]=P_q_charges_knowing_size(q,s,T=T) for q in R_charge]

    # create a set of particle mobility density, one for each number of elementary charges
    u_q_k = zeros(Cdouble,abs(Nq),length(u_imp));
    [u_q_k[abs(q),:] = Pqs[abs(q),:].*u_imp for q in R_charge]

    # return the estimated
    R_charge,K,u_q_k,Pqs
end

"""
    DMA_size_density(u_q_k::Array{Cdouble,2},K::Array{Cdouble,2},s::Array{Cdouble,1},k_meas::Array{Cdouble,1},sig_k_meas::Array{Cdouble,1})

    non-diffusing DMA simulation

    input 

     - u_q_k:      particle mobility density [# m^{-3} m^{-2} V s] keeping track of the charging probabilities (row q is the mobility density for q charges varying with particle diameter)
     - K:          electrical mobility [m^2 V^{-1} s^{-1}] discretization nodes mapping u_q_k
     - s:          unused diameter discretization nodes corresponding to the mobilities K        #WARNING obselete, not in use
     - k_meas:     electrical mobility [m^2 V^{-1} s^{-1}] centroids of the DMA
     - sig_k_meas: electrical mobility spread [m^2 V^{-1} s^{-1}] of the DMA                     #WARNING: obselete, not in use

    output:
     - particle size density [# m^{-3} m^{-1}] right after the DMA 
"""
function DMA_size_density(u_q_k::Array{Cdouble,2},K::Array{Cdouble,2},s::Array{Cdouble,1},k_meas::Array{Cdouble,1},sig_k_meas::Array{Cdouble,1};
    TF_model::String="",Qa::Cdouble=0.3,Qs::Cdouble=0.3,Qc::Cdouble=3.0,Qm::Cdouble=3.0)
    # particle size density (keeping track of the number of charge)
    u_dma_i_q_k = zeros(Cdouble,length(k_meas),size(K,1),size(K,2))
    # transfer function of the DMA (in terms of electrical mobility)
    Psi_dma = zeros(Cdouble,size(K,2));
    for i in 1:length(k_meas) # channel selection
        for q in 1:size(K,1)     # number of charge
            # transfer function for the centroid k_meas[i] for q charges 
            if (TF_model=="Stolzenburg_NDTF") # non-diffusing transfer function 
                global Psi_dma = NDTF.(K[q,:],k_meas[i],Qa,Qs,Qc,Qm)
            elseif (TF_model=="Stolzenburg_DTF") # diffusing transfer function 
                # TODO: implement the actual 
                Gdma = 0.0 # Gdma(R1,R2,r,Qa,Qc,Qt)
                D = 0.0
                global Psi_dma = DTF(K[q,:],k_meas[i],Qa,Qs,Qc,Qm,Gdma,D)
            else # default case is the triangle
                global Psi_dma = triangle_TF.(K[q,:],k_meas[i],abs.(0.5*((Qa+Qs)/(Qc+Qm))*k_meas[i]))
            end

            # discretization of the integral for each number of charge q and centroid k_meas[i] ∫ Ψ_i(k) u_q(k) dk
            u_dma_i_q_k[i,q,:] = Psi_dma.*u_q_k[q,:] # this is for the set of mobility K[q,:]
        end
    end
    # size distribution accounting for multiple charging
    dropdims(sum(u_dma_i_q_k,dims=2),dims=2)
end

"""
    DMA_3010(u::Array{Cdouble,1},s::Array{Cdouble,1},k_meas::Array{Cdouble,1},sig_k_meas::Array{Cdouble,1};
        s50::Cdouble=1.0e-6,delta50::Cdouble=0.1e-6,
        T::Cdouble=293.0,Pr::Cdouble=Pr0,Nq::Int64=6)

    non-diffusing DMA transfer function with impactor and charge conditioner (neutralizer Kr85)

    input 

      - u:               particle size density [# m^{-3} m^{-1}] at the inlet
      - s:               diameter [m] discretization nodes
      - k_meas:          electrical mobility [m^2 V^{-1} s^{-1}] centroids of the DMA
      - sig_k_meas:      electrical mobility spread [m^2 V^{-1} s^{-1}] of the DMA 
      - s50 and delta50: the impactor 50% threshold [m] and sharpness [m]
      - T and Pr:        the temperature [K] and pressure [Pa] at which the device operates
      - Nq:              the signed maximal number of chargers acquired by a particle

    output:
     - particle size density [# m^{-3} m^{-1}] right after the DMA 
"""
function DMA_3010(u::Array{Cdouble,1},s::Array{Cdouble,1},k_meas::Array{Cdouble,1},sig_k_meas::Array{Cdouble,1};
    s50::Cdouble=1.0e-6,delta50::Cdouble=0.1e-6,
    T::Cdouble=293.0,Pr::Cdouble=Pr0,Nq::Int64=6,
    TF_model::String="",Qa::Cdouble=0.3,Qs::Cdouble=0.3,Qc::Cdouble=3.0,Qm::Cdouble=3.0)

    # cut off size by impactor
    u_imp = impactor(u,s;s50=s50,delta50=delta50)
    # charging the particle in the neutralizer
    _,K,u_q_k = neutralizer_Kr_85(u_imp,s;T=T,Pr=Pr,Nq=Nq)
    # mobility selection in the classifier: size density at the output of the mobility analyzer
    DMA_size_density(u_q_k,K,s,k_meas,sig_k_meas;TF_model=TF_model,Qa=Qa,Qs=Qs,Qc=Qc,Qm=Qm)
end