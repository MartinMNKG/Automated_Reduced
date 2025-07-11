import warnings
import numpy as np 
import cantera as ct 
import os 
import sys
def make_dir(dir): 
    os.makedirs(os.path.join(dir,"mech"))
    os.makedirs(os.path.join(dir,"hist"))
    
def write_yaml(gas, filename):
    t_writer = ct.YamlWriter()
    t_writer.add_solution(gas)
    t_writer.to_file(filename)
    return   

def get_factor_dim_ln(t_gas):
    init_value = []
    species = t_gas.species()
    reactions = t_gas.reactions()
    _gas = ct.Solution(
        thermo="IdealGas", kinetics="GasKinetics", species=species, reactions=reactions
    )
    rxns_orig = _gas.reactions()
    p = 0
    for k in range(_gas.n_reactions):
        t_rxn = rxns_orig[k]
        # type_rxns = gas.reaction_type(k)
        type_rxns = t_rxn.reaction_type
        # print(type_rxns)
        if type_rxns == "Arrhenius":
            rate_a = t_rxn.rate.pre_exponential_factor
            
            p = p + 1
            rate_b = t_rxn.rate.temperature_exponent
            
            p = p + 1
            rate_e = t_rxn.rate.activation_energy
            p = p + 1
            init_value.append(np.log(rate_a))
            init_value.append(rate_b)
            init_value.append(rate_e)
        elif type_rxns == "pressure-dependent-Arrhenius":
            t_rates = t_rxn.rate.rates
            for m in range(len(t_rates)):
                t_pres = t_rates[m][0] / 101325.0
                t_rate = t_rates[m][1]
                rate_a = t_rate.pre_exponential_factor
                p = p + 1
                rate_b = t_rate.temperature_exponent
                p = p + 1
                rate_e = t_rate.activation_energy
                p = p + 1
                init_value.append(np.log(rate_a))
                init_value.append(rate_b)
                init_value.append(rate_e)
        elif type_rxns == "three-body-Arrhenius":
            rate_a = t_rxn.rate.pre_exponential_factor
            p = p + 1
            rate_b = t_rxn.rate.temperature_exponent
            p = p + 1
            rate_e = t_rxn.rate.activation_energy
            p = p + 1
            init_value.append(np.log(rate_a))
            init_value.append(rate_b)
            init_value.append(rate_e)
        elif type_rxns == "falloff-Troe" or type_rxns == "falloff-Lindemann":
            lowrate_a = t_rxn.rate.low_rate.pre_exponential_factor
            p = p + 1
            lowrate_b = t_rxn.rate.low_rate.temperature_exponent
            p = p + 1
            lowrate_e = t_rxn.rate.low_rate.activation_energy
            p = p + 1
            highrate_a = t_rxn.rate.high_rate.pre_exponential_factor
            p = p + 1
            highrate_b = t_rxn.rate.high_rate.temperature_exponent
            p = p + 1
            highrate_e = t_rxn.rate.high_rate.activation_energy
            p = p + 1
            init_value.append(np.log(lowrate_a))
            init_value.append(lowrate_b)
            init_value.append(lowrate_e)
            init_value.append(np.log(highrate_a))
            init_value.append(highrate_b)
            init_value.append(highrate_e)
        else:
            warnings.warn("Unsupported reaction type " + type_rxns + ".")
    return p,init_value


def rxns_yaml_arr_list2_ln(t_gas, factor):
    # print(factor)
    species = t_gas.species()
    reactions = t_gas.reactions()
    _gas = ct.Solution(
        thermo="IdealGas", kinetics="GasKinetics", species=species, reactions=reactions
    )
    rxns_orig = _gas.reactions()
    rxns_modd = []
    p = 0
    for k in range(_gas.n_reactions):
        t_rxn = rxns_orig[k]
        str_equ = t_rxn.equation
        # type_rxns = gas.reaction_type(k)
        type_rxns = t_rxn.reaction_type
        t_dup = t_rxn.duplicate
        str_dup = ""
        if t_dup:
            str_dup = ",\nduplicate: true"
        if type_rxns == "Arrhenius":
            
            rate_a =  np.exp(factor[p])
            p = p + 1
            rate_b =  factor[p]
            p = p + 1
            rate_e =  factor[p]
            p = p + 1
            str_rate = (
                "{A: "
                + str(rate_a)
                + ", b: "
                + str(rate_b)
                + ", Ea: "
                + str(rate_e)
                + "}"
            )
            str_cti = (
                "{equation: "
                + str_equ
                + ",\n"
                + "rate-constant: "
                + str_rate
                + str_dup
                + "}"
            )
        elif type_rxns == "pressure-dependent-Arrhenius":
            t_rates = t_rxn.rate.rates
            str_rate = ""
            for m in range(len(t_rates)):
                t_pres = t_rates[m][0] / 101325.0
                t_rate = t_rates[m][1]
                rate_a =  np.exp(factor[p])
                p = p + 1
                rate_b =  factor[p]
                p = p + 1
                rate_e =  factor[p]
                p = p + 1
                str_rate = str_rate + (
                    "{P: "
                    + str(t_pres)
                    + " atm, A: "
                    + str(rate_a)
                    + ", b: "
                    + str(rate_b)
                    + ", Ea: "
                    + str(rate_e)
                    + "},\n"
                )
            str_cti = (
                "{equation: "
                + str_equ
                + ",\n"
                + "type: pressure-dependent-Arrhenius"
                + str_dup
                + ",\n"
                + "rate-constants: \n["
                + str_rate
                + "]\n"
                + "}"
            )
        elif type_rxns == "three-body-Arrhenius":
            str_eff = str(t_rxn.third_body.efficiencies)
            str_eff = str_eff.replace("'", "")
            
            rate_a =  np.exp(factor[p])
            p = p + 1
            rate_b =  factor[p]
            p = p + 1
            rate_e =  factor[p]
            p = p + 1
            str_rate = "[" + str(rate_a) + "," + str(rate_b) + "," + str(rate_e) + "]"
            str_cti = (
                "{equation: "
                + str_equ
                + ",\n"
                + "type: three-body,\n"
                + "rate-constant: "
                + str_rate
                + ",\n"
                + "efficiencies: "
                + str_eff
                + str_dup
                + "}"
            )
            # print(idx)
            # print(str_cti)
            # return ct.Reaction.fromCti(str_cti)
        elif type_rxns == "falloff-Troe" or type_rxns == "falloff-Lindemann":
            # str_type_falloff = t_rxn.falloff.falloff_type
            array_para_falloff = t_rxn.rate.falloff_coeffs
            str_eff = str(t_rxn.third_body.efficiencies)
            str_eff = str_eff.replace("'", "")
            
            lowrate_a =  np.exp(factor[p])
            p = p + 1
            lowrate_b =  factor[p]
            p = p + 1
            lowrate_e =  factor[p]
            p = p + 1
            highrate_a =  np.exp(factor[p])
            p = p + 1
            highrate_b =  factor[p]
            p = p + 1
            highrate_e =  factor[p]
            p = p + 1
            str_lowrate = (
                "{A: "
                + str(lowrate_a)
                + ", b: "
                + str(lowrate_b)
                + ", Ea: "
                + str(lowrate_e)
                + "}"
            )
            str_highrate = (
                "{A: "
                + str(highrate_a)
                + ", b: "
                + str(highrate_b)
                + ", Ea: "
                + str(highrate_e)
                + "}"
            )
            str_cti = (
                "{equation: "
                + str_equ
                + ",\n"
                + "type: falloff,\n"
                + "low-P-rate-constant: "
                + str_lowrate
                + ",\n"
                + "high-P-rate-constant: "
                + str_highrate
                + ",\n"
            )
            if type_rxns == "falloff-Troe":
                if len(array_para_falloff) == 4:
                    str_cti = (
                        str_cti
                        + "Troe: {"
                        + "A: "
                        + str(array_para_falloff[0])
                        + ", T3: "
                        + str(array_para_falloff[1])
                        + ", T1: "
                        + str(array_para_falloff[2])
                        + ", T2: "
                        + str(array_para_falloff[3])
                        + "},\n"
                    )
                elif len(array_para_falloff) == 3:
                    str_cti = (
                        str_cti
                        + "Troe: {"
                        + "A: "
                        + str(array_para_falloff[0])
                        + ", T3: "
                        + str(array_para_falloff[1])
                        + ", T1: "
                        + str(array_para_falloff[2])
                        + "},\n"
                    )
            str_cti = str_cti + "efficiencies: " + str_eff + str_dup + "}"
            # print(str_cti)
        tt_rxn = ct.Reaction.from_yaml(str_cti, _gas)
        rxns_modd.append(tt_rxn)
    _gas = ct.Solution(
        thermo="IdealGas", kinetics="GasKinetics", species=species, reactions=rxns_modd
    )
    return _gas

