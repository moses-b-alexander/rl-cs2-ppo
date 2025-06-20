


"""IMPORTS"""


from bs4 import BeautifulSoup
from copy import deepcopy
from demoparser2 import DemoParser
from fake_useragent import UserAgent
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import numpy as np
import os
from pandas import DataFrame
from pandas import concat, factorize, get_dummies, read_pickle, set_option
from patoolib import extract_archive
import pickle
import psutil as ps
import ray
from ray import air, tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig as RLConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
import re
import requests
import shutil
from time import perf_counter, sleep
import torch
import torch.nn as nn
from uuid import uuid4
import warnings


"""VARIABLES"""


root_dir = os.getcwd()
assets_dir = os.path.join(root_dir, "assets")
data_dir = os.path.join(root_dir, "demo")
logs_dir = os.path.join(root_dir, "logs")
logs_ct_dir = os.path.join(root_dir, "logs_ct")
logs_t_dir = os.path.join(root_dir, "logs_t")
frame_dir = os.path.join(root_dir, "frame")
npz_dir = os.path.join(root_dir, "npz")
cloud_dir = os.path.join(root_dir, "cloud")
if not os.path.exists(logs_dir):  os.makedirs(logs_dir)
if not os.path.exists(logs_ct_dir):  os.makedirs(logs_ct_dir)
if not os.path.exists(logs_t_dir):  os.makedirs(logs_t_dir)
if not os.path.exists(frame_dir):  os.makedirs(frame_dir)
if not os.path.exists(npz_dir):  os.makedirs(npz_dir)
if not os.path.exists(cloud_dir):  os.makedirs(cloud_dir)


warnings.filterwarnings('ignore', category=DeprecationWarning)


set_option("display.max_rows", 100000)

set_option("display.max_columns", 1000)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


num_cpus = ps.cpu_count(logical=False) - 2

num_gpus = \
    len([torch.cuda.device(g) for g in range(torch.cuda.device_count())])


seed = 1

epsilon = 10 ** -6

pause_time = 10.0

num_threads = 1000

match_count_limit = 64
tick_limit = 317440
row_limit = tick_limit * 10

tick_count = 64
tick_rate = 1 / tick_count

minimum_round_duration = 5
maximum_round_duration = 160

player_count = 1
team_count = 5
t_team_count, ct_team_count = 5, 5
players_count = t_team_count + ct_team_count

assist_factor = 0.5

ct_defused_factor = 1.0
ct_delayed_factor = 1.0
t_exploded_factor = 1.0

ct_eliminated_factor = 1.0
t_eliminated_factor = 1.0


bnd_is_alive = False


str_str = "string"

cs2_str, valve_str = "CS2", "Valve"

hltv_url_str, hltv_str = "https://www.hltv.org", "HLTV"

t_str, vs_str, ct_str = "T", "_vs_", "CT"

old_round_start_str, old_round_end_str = "cs_round_final_beep", "round_end"
round_start_str, round_end_str = "astart_of_round", "aend_of_round"

initial_type_str, terminal_type_str = "a0", "a1"

dt_fmt_str = "%Y-%m-%d|%H:%M:%S.%f"

pickle_str = "pkl"
npz_str = "npz"


active_map_names = [
    "ancient", "anubis", "dust2", "inferno",
    "mirage", "nuke", "train",
    "overpass", "vertigo",
]


map_offsets = {}

map_offsets["ancient"] = {
    "scale": 5.00,
    "pos_x": -2953,
    "pos_y": 2164,
    "CTSpawn_x": 0.510,
    "CTSpawn_y": 0.170,
    "TSpawn_x": 0.485,
    "TSpawn_y": 0.870,
    "inset_left": 0.000,
    "inset_top": 0.000,
    "inset_right": 0.000,
    "inset_bottom": 0.000,
    "rotate": 0.0,
    "zoom": 0.0,
}

map_offsets["anubis"] = {
    "scale": 5.22,
    "pos_x": -2796,
    "pos_y": 3328,
    "CTSpawn_x": 0.610,
    "CTSpawn_y": 0.220,
    "TSpawn_x": 0.580,
    "TSpawn_y": 0.930,
    "inset_left": 0.000,
    "inset_top": 0.000,
    "inset_right": 0.000,
    "inset_bottom": 0.000,
    "rotate": 0.0,
    "zoom": 0.0,
}

map_offsets["dust2"] = {
    "scale": 4.40,
    "pos_x": -2476,
    "pos_y": 3239,
    "CTSpawn_x": 0.620,
    "CTSpawn_y": 0.210,
    "TSpawn_x": 0.390,
    "TSpawn_y": 0.910,
    "inset_left": 0.000,
    "inset_top": 0.000,
    "inset_right": 0.000,
    "inset_bottom": 0.000,
    "rotate": 1.0,
    "zoom": 1.1,
}

map_offsets["inferno"] = {
    "scale": 4.90,
    "pos_x": -2087,
    "pos_y": 3870,
    "CTSpawn_x": 0.900,
    "CTSpawn_y": 0.350,
    "TSpawn_x": 0.100,
    "TSpawn_y": 0.670,
    "inset_left": 0.000,
    "inset_top": 0.000,
    "inset_right": 0.000,
    "inset_bottom": 0.000,
    "rotate": 0.0,
    "zoom": 0.0,
}

map_offsets["mirage"] = {
    "scale": 5.00,
    "pos_x": -3230,
    "pos_y": 1713,
    "CTSpawn_x": 0.280,
    "CTSpawn_y": 0.700,
    "TSpawn_x": 0.870,
    "TSpawn_y": 0.360,
    "inset_left": 0.135,
    "inset_top": 0.080,
    "inset_right": 0.105,
    "inset_bottom": 0.080,
    "rotate": 0.0,
    "zoom": 0.0,
}

map_offsets["nuke"] = {
    "scale": 7.00,
    "pos_x": -3453,
    "pos_y": 2887,
    "CTSpawn_x": 0.820,
    "CTSpawn_y": 0.450,
    "TSpawn_x": 0.190,
    "TSpawn_y": 0.540,
    "inset_left": 0.330,
    "inset_top": 0.200,
    "inset_right": 0.200,
    "inset_bottom": 0.200,
    "rotate": 0.0,
    "zoom": 0.0,
}

map_offsets["overpass"] = {
    "scale": 5.20,
    "pos_x": -4831,
    "pos_y": 1781,
    "CTSpawn_x": 0.490,
    "CTSpawn_y": 0.200,
    "TSpawn_x": 0.660,
    "TSpawn_y": 0.930,
    "inset_left": 0.000,
    "inset_top": 0.000,
    "inset_right": 0.000,
    "inset_bottom": 0.000,
    "rotate": 0.0,
    "zoom": 0.0,
}

map_offsets["vertigo"] = {
    "scale": 4.00,
    "pos_x": -3168,
    "pos_y": 1762,
    "CTSpawn_x": 0.540,
    "CTSpawn_y": 0.250,
    "TSpawn_x": 0.200,
    "TSpawn_y": 0.750,
    "inset_left": 0.100,
    "inset_top": 0.100,
    "inset_right": 0.200,
    "inset_bottom": 0.150,
    "rotate": 0.0,
    "zoom": 0.0,
}


weapon_names = [
    "AK-47", "AUG", "AWP", "C4", "CZ75-Auto", "Decoy Grenade",
    "Desert Eagle", "Dual Berettas", "FAMAS", "Five-SeveN", "Flashbang",
    "G3SG1", "Galil AR", "Glock-18", "High Explosive Grenade",
    "Incendiary Grenade", "Knife", "M249", "M4A1-S", "M4A4", "MAC-10",
    "MAG-7", "MP5-SD", "MP7", "MP9", "Molotov", "Negev", "nnnnn",
    "Nova", "P2000", "P250", "P90", "PP-Bizon", "R8 Revolver", "SCAR-20",
    "SG 553", "SSG 08", "Sawed-Off", "Smoke Grenade", "Tec-9", "UMP-45",
    "USP-S", "XM1014", "Zeus x27",
]

weapon_types = {
    "AK-47": "rifle", "AUG": "rifle", "AWP": "sniper", "C4": "c4",
    "CZ75-Auto": "pistol", "Decoy Grenade": "grenade",
    "Desert Eagle": "pistol", "Dual Berettas": "pistol", "FAMAS": "rifle",
    "Five-SeveN": "pistol", "Flashbang": "grenade", "G3SG1": "sniper",
    "Galil AR": "rifle", "Glock-18": "pistol",
    "High Explosive Grenade": "grenade", "Incendiary Grenade": "grenade",
    "Knife": "knife", "M249": "lmg", "M4A1-S": "rifle", "M4A4": "rifle",
    "MAC-10": "smg", "MAG-7": "shotgun", "MP5-SD": "smg", "MP7": "smg",
    "MP9": "smg", "Molotov": "grenade", "Negev": "lmg", "nnnnn": "nnnnn",
    "Nova": "shotgun", "P2000": "pistol", "P250": "pistol", "P90": "smg",
    "PP-Bizon": "smg", "R8 Revolver": "pistol", "SCAR-20": "sniper",
    "SG 553": "rifle", "SSG 08": "sniper", "Sawed-Off": "shotgun",
    "Smoke Grenade": "grenade", "Tec-9": "pistol", "UMP-45": "smg",
    "USP-S": "pistol", "XM1014": "shotgun", "Zeus x27": "knife",
}


grenade_types = [
    "grenade_blaze", "grenade_frag", "grenade_decoy",
    "grenade_flashbang", "grenade_smoke",
]


action_buttons = [
    "fire", "reload", "use",
    "forward", "left", "right", "back",
]


tick_attrs = [
    "player_name", "player_steamid",
    "team_num",
    "has_defuser",
    "health", "armor_value", "has_helmet", "is_alive",
    "score",
    "X", "Y", "Z",
    "velocity_X", "velocity_Y", "velocity_Z",
    "velocity", "velo_modifier", "pitch", "yaw", "fall_velo",
    "last_place_name",
    "spawn_time", "death_time",
    "game_time",
    "balance", "start_balance",
    "cash_spent_this_round", "current_equip_value",
    "move_type",
    "spotted", "approximate_spotted_by",
    "old_jump_pressed",
    "duck_amount", "ducking",
    "is_walking", "is_scoped", "is_defusing",
    "moved_since_spawn",
    "in_bomb_zone", "in_buy_zone", "in_no_defuse_area",
    "flash_duration", "flash_max_alpha",
    "time_last_injury",
    "shots_fired",
    "ping",
    "is_connected",
    "weapon_mode",
    "zoom_lvl", "accuracy_penalty",
    "active_weapon_name",
    "active_weapon_ammo", "total_ammo_left",
    "is_silencer_on", "is_burst_mode", "is_in_reload",
    "burst_shots_remaining",
    "active_weapon_original_owner",
    "orig_team_number",
    "last_shot_time",
    "team_surrendered",
    "team_score_first_half", "team_score_overtime", "team_score_second_half",
    "is_warmup_period", "is_freeze_period", "is_technical_timeout",
    "is_waiting_for_resume",
    "is_terrorist_timeout", "is_ct_timeout",
    "num_terrorist_timeouts", "num_ct_timeouts",
    "is_bomb_dropped", "is_bomb_planted",
    "round_win_status", "round_win_reason",
    "ct_losing_streak", "t_losing_streak",
    "kills_total", "deaths_total", "assists_total", "headshot_kills_total",
    "damage_total", "utility_damage_total", "enemies_flashed_total",
    "objective_total",
    "game_phase", "total_rounds_played", "rounds_played_this_phase",
    "team_name", "team_clan_name",
]


total_int_stats = [
    "starting_cash_total", "cash_spent_total", "headshot_kills_total",
    "kills_total", "deaths_total", "assists_total", "kd_spread_total",
    "damage_total", "utility_damage_total", "flashes_total",
    "x_d_total", "y_d_total", "z_d_total",
    "objective_total", "alive_total",
]
total_ratio_stats = [
    "kd_ratio_total", "headshot_ratio_total", "utility_ratio_total",
]

phase_int_stats = [c.replace("total", "phase") for c in total_int_stats]
phase_ratio_stats = [c.replace("total", "phase") for c in total_ratio_stats]


total_per_round_stats = \
    [c.replace("total", "per_round_total")
    for c in total_int_stats + total_ratio_stats if "alive" not in c]
total_per_second_stats = \
    [c.replace("total", "per_second_total")
    for c in total_int_stats + total_ratio_stats if "alive" not in c]

phase_per_round_stats = \
    [c.replace("phase", "per_round_phase")
    for c in phase_int_stats + phase_ratio_stats if "alive" not in c]
phase_per_second_stats = \
    [c.replace("phase", "per_second_phase")
    for c in phase_int_stats + phase_ratio_stats if "alive" not in c]


grenade_counter = \
    {"blaze": 0, "frag": 0, "decoy": 0, "flashbang": 0, "smoke": 0}


old_event_types = [
    initial_type_str, round_start_str,
    "bomb_planted", "bomb_defused", "bomb_exploded",
    "blaze_detonate", "frag_detonate",
    "decoy_detonate", "flash_detonate", "smoke_detonate",
    "player_sound", "weapon_fire",
    "player_hurt", "player_death", "player_disconnect",
    round_end_str, terminal_type_str,
]

event_types = [
    initial_type_str, round_start_str,
    "audio_footsteps", "audio_gunfire",
    "grenade_blaze", "grenade_frag",
    "grenade_decoy", "grenade_flashbang", "grenade_smoke",
    "objective_planted", "objective_defused", "objective_exploded",
    "visible_damage", "visible_death", "visible_disconnect",
    round_end_str, terminal_type_str,
]

event_rename_types = {
    initial_type_str: initial_type_str, round_start_str: round_start_str,
    "player_sound": "audio_footsteps",
    "weapon_fire": "audio_gunfire",
    "blaze_detonate": "grenade_blaze",
    "frag_detonate": "grenade_frag",
    "decoy_detonate": "grenade_decoy",
    "flash_detonate": "grenade_flashbang",
    "smoke_detonate": "grenade_smoke",
    "bomb_planted": "objective_planted",
    "bomb_defused": "objective_defused",
    "bomb_exploded": "objective_exploded",
    "player_hurt": "visible_damage",
    "player_death": "visible_death",
    "player_disconnect": "visible_disconnect",
    round_end_str: round_end_str, terminal_type_str: terminal_type_str,
}


cols = {
    "node": {
        "situation": {
            "accuracy_penalty": "float32",
            "armor_value": "uint8",
            "back": "bool",
            "cash_value": "int32",
            "duck_amount": "float32",
            "equipment_value": "int32",
            "fire": "bool",
            "forward": "bool",
            "has_defuser": "bool", "has_helmet": "float32",
            "health_value": "uint8",
            "heard_footsteps": "bool", "heard_gunfire": "bool",
            "in_bomb_zone": "bool", "in_buy_zone": "bool",
            "in_no_defuse_area": "bool",
            "is_alive": "bool",
            "is_burned": "bool",
            "is_burst_mode": "bool",
            "is_defusing": "bool",
            "is_ducking": "bool",
            "is_falling": "bool",
            "is_flashed": "bool",
            "is_reloading": "bool",
            "is_scoping": "bool",
            "is_silencing": "bool",
            "is_walking": "bool",
            "lane_a_x_distance": "float32",
            "lane_a_y_distance": "float32",
            "lane_a_z_distance": "float32",
            "lane_b_x_distance": "float32",
            "lane_b_y_distance": "float32",
            "lane_b_z_distance": "float32",
            "lane_m_x_distance": "float32",
            "lane_m_y_distance": "float32",
            "lane_m_z_distance": "float32",
            "lane_a_primary": "bool", "lane_a_secondary": "bool",
            "lane_b_primary": "bool", "lane_b_secondary": "bool",
            "lane_m_primary": "bool", "lane_m_secondary": "bool",
            "left": "bool",
            "moving_default": "bool", "moving_ladder": "bool",
            "is_jumping": "bool",
            "reload": "bool",
            "right": "bool",
            "seen": "bool",
            "use": "bool",
            "pitch": "float32", "yaw": "float32",
            "weapon_ammo": "uint8",
            "weapon_ammo_fired": "uint8", "weapon_ammo_remaining": "uint8",
            "weapon_bought_by_opponent": "bool",
            "weapon_bought_by_self": "bool",
            "weapon_bought_by_teammate": "bool",
            "weapon_type_c4": "bool",
            "weapon_type_grenade": "bool",
            "weapon_type_knife": "bool",
            "weapon_type_lmg": "bool",
            "weapon_type_nnnnn": "bool",
            "weapon_type_pistol": "bool",
            "weapon_type_rifle": "bool",
            "weapon_type_shotgun": "bool",
            "weapon_type_smg": "bool",
            "weapon_type_sniper": "bool",
            "weapon_name_ak-47": "bool",
            "weapon_name_aug": "bool",
            "weapon_name_awp": "bool",
            "weapon_name_blaze-grenade": "bool",
            "weapon_name_c4": "bool",
            "weapon_name_cz75-auto": "bool",
            "weapon_name_decoy-grenade": "bool",
            "weapon_name_deserteagle": "bool",
            "weapon_name_dualberettas": "bool",
            "weapon_name_famas": "bool",
            "weapon_name_five-seven": "bool",
            "weapon_name_flashbang-grenade": "bool",
            "weapon_name_frag-grenade": "bool",
            "weapon_name_g3sg1": "bool",
            "weapon_name_galilar": "bool",
            "weapon_name_glock-18": "bool",
            "weapon_name_knife": "bool",
            "weapon_name_m249": "bool",
            "weapon_name_m4a1-s": "bool",
            "weapon_name_m4a4": "bool",
            "weapon_name_mac-10": "bool",
            "weapon_name_mag-7": "bool",
            "weapon_name_mp5-sd": "bool",
            "weapon_name_mp7": "bool",
            "weapon_name_mp9": "bool",
            "weapon_name_negev": "bool",
            "weapon_name_nnnnn": "bool",
            "weapon_name_nova": "bool",
            "weapon_name_p2000": "bool",
            "weapon_name_p250": "bool",
            "weapon_name_p90": "bool",
            "weapon_name_pp-bizon": "bool",
            "weapon_name_r8revolver": "bool",
            "weapon_name_scar-20": "bool",
            "weapon_name_sg553": "bool",
            "weapon_name_ssg08": "bool",
            "weapon_name_sawed-off": "bool",
            "weapon_name_smoke-grenade": "bool",
            "weapon_name_tec-9": "bool",
            "weapon_name_ump-45": "bool",
            "weapon_name_usp-s": "bool",
            "weapon_name_xm1014": "bool",
            "weapon_name_zeusx27": "bool",
            "x_p": "float32", "x_d": "float32", "x_v": "float32",
            "y_p": "float32", "y_d": "float32", "y_v": "float32",
            "z_p": "float32", "z_d": "float32", "z_v": "float32",
            "zoom_level": "float32",
        },
        "round": {
            "alive_per_round_phase": "float32",
            "alive_per_round_total": "float32",
            "alive_phase": "int32", "alive_total": "int32",
            "assists_per_round_phase": "float32",
            "assists_per_round_total": "float32",
            "assists_per_second_phase": "float32",
            "assists_per_second_total": "float32",
            "assists_phase": "uint8", "assists_total": "uint8",
            "cash_spent_per_round_phase": "float32",
            "cash_spent_per_round_total": "float32",
            "cash_spent_per_second_phase": "float32",
            "cash_spent_per_second_total": "float32",
            "cash_spent_phase": "int32", "cash_spent_total": "int32",
            "damage_per_round_phase": "float32",
            "damage_per_round_total": "float32",
            "damage_per_second_phase": "float32",
            "damage_per_second_total": "float32",
            "damage_phase": "int32", "damage_total": "int32",
            "deaths_per_round_phase": "float32",
            "deaths_per_round_total": "float32",
            "deaths_per_second_phase": "float32",
            "deaths_per_second_total": "float32",
            "deaths_phase": "uint8", "deaths_total": "uint8",
            "x_d_per_round_phase": "float32",
            "x_d_per_round_total": "float32",
            "x_d_per_second_phase": "float32",
            "x_d_per_second_total": "float32",
            "x_d_phase": "float32", "x_d_total": "float32",
            "y_d_per_round_phase": "float32",
            "y_d_per_round_total": "float32",
            "y_d_per_second_phase": "float32",
            "y_d_per_second_total": "float32",
            "y_d_phase": "float32", "y_d_total": "float32",
            "z_d_per_round_phase": "float32",
            "z_d_per_round_total": "float32",
            "z_d_per_second_phase": "float32",
            "z_d_per_second_total": "float32",
            "z_d_phase": "float32", "z_d_total": "float32",
            "flashes_per_round_phase": "float32",
            "flashes_per_round_total": "float32",
            "flashes_per_second_phase": "float32",
            "flashes_per_second_total": "float32",
            "flashes_phase": "uint8", "flashes_total": "uint8",
            "headshot_kills_per_round_phase": "float32",
            "headshot_kills_per_round_total": "float32",
            "headshot_kills_per_second_phase": "float32",
            "headshot_kills_per_second_total": "float32",
            "headshot_kills_phase": "uint8",
            "headshot_kills_total": "uint8",
            "headshot_ratio_per_round_phase": "float32",
            "headshot_ratio_per_round_total": "float32",
            "headshot_ratio_per_second_phase": "float32",
            "headshot_ratio_per_second_total": "float32",
            "headshot_ratio_phase": "float32",
            "headshot_ratio_total": "float32",
            "is_ct": "bool", "is_t": "bool",
            "kd_ratio_per_round_phase": "float32",
            "kd_ratio_per_round_total": "float32",
            "kd_ratio_per_second_phase": "float32",
            "kd_ratio_per_second_total": "float32",
            "kd_ratio_phase": "float32", "kd_ratio_total": "float32",
            "kd_spread_per_round_phase": "float32",
            "kd_spread_per_round_total": "float32",
            "kd_spread_per_second_phase": "float32",
            "kd_spread_per_second_total": "float32",
            "kd_spread_phase": "uint8", "kd_spread_total": "uint8",
            "kills_per_round_phase": "float32",
            "kills_per_round_total": "float32",
            "kills_per_second_phase": "float32",
            "kills_per_second_total": "float32",
            "kills_phase": "uint8", "kills_total": "uint8",
            "objectives_per_round_phase": "float32",
            "objectives_per_round_total": "float32",
            "objectives_per_second_phase": "float32",
            "objectives_per_second_total": "float32",
            "objectives_phase": "uint8", "objectives_total": "uint8",
            "starting_cash_per_round_phase": "float32",
            "starting_cash_per_round_total": "float32",
            "starting_cash_per_second_phase": "float32",
            "starting_cash_per_second_total": "float32",
            "starting_cash_phase": "int32",
            "starting_cash_total": "int32",
            "utility_damage_per_round_phase": "float32",
            "utility_damage_per_round_total": "float32",
            "utility_damage_per_second_phase": "float32",
            "utility_damage_per_second_total": "float32",
            "utility_damage_phase": "int32",
            "utility_damage_total": "int32",
            "utility_ratio_per_round_phase": "float32",
            "utility_ratio_per_round_total": "float32",
            "utility_ratio_per_second_phase": "float32",
            "utility_ratio_per_second_total": "float32",
            "utility_ratio_phase": "float32",
            "utility_ratio_total": "float32",
        },
        "game": {
        },
        "match": {
            "player_name": "string", "player_steamid": "string",
            "player_team": "string",
        },
    },
    "downsample": {
        "situation": {
            "end_of_match": "bool",
            "end_of_phase": "bool",
            "end_of_round": "bool",
            "is_bomb_dropped": "bool", "is_bomb_planted": "bool",
            "num_blazes": "uint8",
            "num_blazes_a": "uint8",
            "num_blazes_m": "uint8",
            "num_blazes_b": "uint8",
            "num_decoys": "uint8",
            "num_decoys_a": "uint8",
            "num_decoys_m": "uint8",
            "num_decoys_b": "uint8",
            "num_flashbangs": "uint8",
            "num_flashbangs_a": "uint8",
            "num_flashbangs_m": "uint8",
            "num_flashbangs_b": "uint8",
            "num_frags": "uint8",
            "num_frags_a": "uint8",
            "num_frags_m": "uint8",
            "num_frags_b": "uint8",
            "num_players_ct": "uint8", "num_players_t": "uint8",
            "num_players_total": "uint8",
            "num_smokes": "uint8",
            "num_smokes_a": "uint8",
            "num_smokes_m": "uint8",
            "num_smokes_b": "uint8",
            "score_first_half_ct": "uint8",
            "score_second_half_ct": "uint8",
            "score_first_half_t": "uint8",
            "score_second_half_t": "uint8",
            "start_of_match": "bool",
            "start_of_phase": "bool",
            "start_of_round": "bool",
            "won_with_ct_defused": "bool", "won_with_ct_delayed": "bool",
            "won_with_ct_eliminated": "bool", "won_with_t_eliminated": "bool",
            "won_with_t_exploded": "bool",
        },
        "round": {
            "first_half": "bool",
            "follows_ct_timeout": "bool",
            "follows_technical_timeout": "bool",
            "follows_t_timeout": "bool",
            "round_phase": "uint8", "round_total": "uint8",
            "second_half": "bool",
        },
        "game": {
        },
        "match": {
        },
    },
    "graph": {
        "situation": {
            "armor": "uint8",
            "cost": "int32",
            "death": "bool",
            "disconnect": "bool",
            "distance": "float32",
            "damage_armor": "int32", "damage_health": "int32",
            "double_penetrated": "uint8",
            "grenade": "bool", "grenade_type": "string",
            "headshot": "bool",
            "health": "uint8",
            "hitgroup_generic": "bool", "hitgroup_head": "bool",
            "hitgroup_chest": "bool", "hitgroup_stomach": "bool",
            "hitgroup_leftarm": "bool", "hitgroup_rightarm": "bool",
            "hitgroup_leftleg": "bool", "hitgroup_rightleg": "bool",
            "hitgroup_gear": "bool",
            "item": "string", "item_name": "string",
            "lane_a": "bool",
            "lane_b": "bool",
            "lane_m": "bool",
            "lane_a_x_loc_distance": "float32",
            "lane_a_y_loc_distance": "float32",
            "lane_a_z_loc_distance": "float32",
            "lane_b_x_loc_distance": "float32",
            "lane_b_y_loc_distance": "float32",
            "lane_b_z_loc_distance": "float32",
            "lane_m_x_loc_distance": "float32",
            "lane_m_y_loc_distance": "float32",
            "lane_m_z_loc_distance": "float32",
            "name": "string",
            "noscope": "bool",
            "not_penetrated": "uint8",
            "single_penetrated": "uint8",
            "silenced": "bool", "silent": "bool",
            "steamid": "string",
            "thrusmoke": "bool",
            "tick": "int32", "timestep": "int32",
            "type": "string",
            "urgent": "bool",
            "user_name": "string", "user_steamid": "string",
            "weapon": "string",
            "x_loc": "float32", "y_loc": "float32", "z_loc": "float32",
        },
        "round": {
        },
        "game": {
            "map_name_ancient": "bool",
            "map_name_anubis": "bool",
            "map_name_dust2": "bool",
            "map_name_inferno": "bool",
            "map_name_mirage": "bool",
            "map_name_nuke": "bool",
            "map_name_overpass": "bool",
            "map_name_vertigo": "bool",
            "map_num": "uint8",
        },
        "match": {
            "match": "string", "server_name": "string", "tournament": "string",
        },
    },
}


def recurse_dict(d):
    for k, v in d.items():
        if type(v) is dict:  yield from recurse_dict(v)
        else:  yield (k, v)

def get_nested_dict(d):
    if not any(isinstance(v, dict) for v in d.values()):
        yield d
    else:
        for v in d.values():
            if isinstance(v, dict):
                yield from get_nested_dict(v)


all_cols = {k: v for k, v in recurse_dict(cols)}


node_cols = {k: v for k, v in recurse_dict(cols["node"])}

downsample_cols = {k: v for k, v in recurse_dict(cols["downsample"])}

graph_cols = {k: v for k, v in recurse_dict(cols["graph"])}


index_cols = {
    "game_idx": "string",
    "tick": "int32",
    "player_steamid": "string",
}


label_cols = {
    "player_steamid": "string",
}


reward_cols = {
    "won_with_ct_defused": "bool",
    "won_with_ct_delayed": "bool",
    "won_with_ct_eliminated": "bool",
    "won_with_t_eliminated": "bool",
    "won_with_t_exploded": "bool",
}


empty_dict = {}


action_movement_dict = {
    "back": "bool",
    "forward": "bool",
    "left": "bool",
    "right": "bool",
}

action_item_dict = {
    "fire": "bool",
    "reload": "bool",
    "use": "bool",
}

action_view_dict = {
    "pitch": "float32",
    "yaw": "float32",
}

action_delta_dict = {
    "x_d": "float32",
    "y_d": "float32",
    "z_d": "float32",
}

action_categorical_dict = {
    "movement": action_movement_dict,
    "item": action_item_dict,
}

action_numerical_dict = {
    "view": action_view_dict,
    # "delta": action_delta_dict,
}

action_dict = {
    "categorical": action_categorical_dict,
    "numerical": action_numerical_dict,
}


is_alive_dict = {"is_alive": "bool", }

debuffs_dict = {
    "is_burned": "bool",
    "is_flashed": "bool",
}

health_dict = {
    "armor_value": "float32",
    "health_value": "float32",
    "has_helmet": "float32",
}

money_dict = {
    "cash_value": "int32",
    "equipment_value": "int32",
}

xyz_kin_dict = {
    "x": {
        "x_kinematic": {
            "x_p": "float32",
            "x_v": "float32",
        },
        "lane_x_distance": {
            "lane_a_x_distance": "float32",
            "lane_b_x_distance": "float32",
            "lane_m_x_distance": "float32",
        },
    },
    "y": {
        "y_kinematic": {
            "y_p": "float32",
            "y_v": "float32",
        },
        "lane_y_distance": {
            "lane_a_y_distance": "float32",
            "lane_b_y_distance": "float32",
            "lane_m_y_distance": "float32",
        },
    },
    "z": {
        "z_kinematic": {
            "z_p": "float32",
            "z_v": "float32",
        },
        "lane_z_distance": {
            "lane_a_z_distance": "float32",
            "lane_b_z_distance": "float32",
            "lane_m_z_distance": "float32",
        },
    },
}

pvlane_kin_dict = {
    "pv": {
        "position": {
            "x_p": "float32",
            "y_p": "float32",
            "z_p": "float32",
        },
        "velocity": {
            "x_v": "float32",
            "y_v": "float32",
            "z_v": "float32",
        },
    },
    "lane": {
        "lane_a": {
            "lane_a_x_distance": "float32",
            "lane_a_y_distance": "float32",
            "lane_a_z_distance": "float32",
        },
        "lane_b": {
            "lane_b_x_distance": "float32",
            "lane_b_y_distance": "float32",
            "lane_b_z_distance": "float32",
        },
        "lane_m": {
            "lane_m_x_distance": "float32",
            "lane_m_y_distance": "float32",
            "lane_m_z_distance": "float32",
        },
    },
}

three_lane_dict = {
    "lane_primary": {
        "lane_a_primary": "bool",
        "lane_b_primary": "bool",
        "lane_m_primary": "bool",
    },
    "lane_secondary": {
        "lane_a_secondary": "bool",
        "lane_b_secondary": "bool",
        "lane_m_secondary": "bool",
    },
}

relevance_lane_dict = {
    "lane_a_relevance": {
        "lane_a_primary": "bool",
        "lane_a_secondary": "bool",
    },
    "lane_b_relevance": {
        "lane_b_primary": "bool",
        "lane_b_secondary": "bool",
    },
    "lane_m_relevance": {
        "lane_m_primary": "bool",
        "lane_m_secondary": "bool",
    },
}

defuser_dict = {
    "has_defuser": "bool",
    "is_defusing": "bool",
}

movement_type_dict = {
    "moving_default": "bool",
    "moving_ladder": "bool",
}

alive_dict = {
    "alive_per_round_phase": "float32",
    "alive_per_round_total": "float32",
    "alive_phase": "int32",
    "alive_total": "int32",
}

assists_dict = {
    "assists_per_round_phase": "float32",
    "assists_per_round_total": "float32",
    "assists_per_second_phase": "float32",
    "assists_per_second_total": "float32",
    "assists_phase": "uint8",
    "assists_total": "uint8",
}

cash_spent_dict = {
    "cash_spent_per_round_phase": "float32",
    "cash_spent_per_round_total": "float32",
    "cash_spent_per_second_phase": "float32",
    "cash_spent_per_second_total": "float32",
    "cash_spent_phase": "int32",
    "cash_spent_total": "int32",
}

damage_dict = {
    "damage_per_round_phase": "float32",
    "damage_per_round_total": "float32",
    "damage_per_second_phase": "float32",
    "damage_per_second_total": "float32",
    "damage_phase": "int32",
    "damage_total": "int32",
}

deaths_dict = {
    "deaths_per_round_phase": "float32",
    "deaths_per_round_total": "float32",
    "deaths_per_second_phase": "float32",
    "deaths_per_second_total": "float32",
    "deaths_phase": "uint8",
    "deaths_total": "uint8",
}

flashes_dict = {
    "flashes_per_round_phase": "float32",
    "flashes_per_round_total": "float32",
    "flashes_per_second_phase": "float32",
    "flashes_per_second_total": "float32",
    "flashes_phase": "uint8",
    "flashes_total": "uint8",
}

headshot_kills_dict = {
    "headshot_kills_per_round_phase": "float32",
    "headshot_kills_per_round_total": "float32",
    "headshot_kills_per_second_phase": "float32",
    "headshot_kills_per_second_total": "float32",
    "headshot_kills_phase": "uint8",
    "headshot_kills_total": "uint8",
}

headshot_ratio_dict = {
    "headshot_ratio_per_round_phase": "float32",
    "headshot_ratio_per_round_total": "float32",
    "headshot_ratio_per_second_phase": "float32",
    "headshot_ratio_per_second_total": "float32",
    "headshot_ratio_phase": "float32",
    "headshot_ratio_total": "float32",
}

kd_ratio_dict = {
    "kd_ratio_per_round_phase": "float32",
    "kd_ratio_per_round_total": "float32",
    "kd_ratio_per_second_phase": "float32",
    "kd_ratio_per_second_total": "float32",
    "kd_ratio_phase": "float32",
    "kd_ratio_total": "float32",
}

kd_spread_dict = {
    "kd_spread_per_round_phase": "float32",
    "kd_spread_per_round_total": "float32",
    "kd_spread_per_second_phase": "float32",
    "kd_spread_per_second_total": "float32",
    "kd_spread_phase": "uint8",
    "kd_spread_total": "uint8",
}

kills_dict = {
    "kills_per_round_phase": "float32",
    "kills_per_round_total": "float32",
    "kills_per_second_phase": "float32",
    "kills_per_second_total": "float32",
    "kills_phase": "uint8",
    "kills_total": "uint8",
}

objectives_dict = {
    "objectives_per_round_phase": "float32",
    "objectives_per_round_total": "float32",
    "objectives_per_second_phase": "float32",
    "objectives_per_second_total": "float32",
    "objectives_phase": "uint8",
    "objectives_total": "uint8",
}

starting_cash_dict = {
    "starting_cash_per_round_phase": "float32",
    "starting_cash_per_round_total": "float32",
    "starting_cash_per_second_phase": "float32",
    "starting_cash_per_second_total": "float32",
    "starting_cash_phase": "int32",
    "starting_cash_total": "int32",
}

utility_damage_dict = {
    "utility_damage_per_round_phase": "float32",
    "utility_damage_per_round_total": "float32",
    "utility_damage_per_second_phase": "float32",
    "utility_damage_per_second_total": "float32",
    "utility_damage_phase": "int32",
    "utility_damage_total": "int32",
}

utility_ratio_dict = {
    "utility_ratio_per_round_phase": "float32",
    "utility_ratio_per_round_total": "float32",
    "utility_ratio_per_second_phase": "float32",
    "utility_ratio_per_second_total": "float32",
    "utility_ratio_phase": "int32",
    "utility_ratio_total": "int32",
}

x_d_dict = {
    "x_d_per_round_phase": "float32",
    "x_d_per_round_total": "float32",
    "x_d_per_second_phase": "float32",
    "x_d_per_second_total": "float32",
    "x_d_phase": "int32",
    "x_d_total": "int32",
}

y_d_dict = {
    "y_d_per_round_phase": "float32",
    "y_d_per_round_total": "float32",
    "y_d_per_second_phase": "float32",
    "y_d_per_second_total": "float32",
    "y_d_phase": "int32",
    "y_d_total": "int32",
}

z_d_dict = {
    "z_d_per_round_phase": "float32",
    "z_d_per_round_total": "float32",
    "z_d_per_second_phase": "float32",
    "z_d_per_second_total": "float32",
    "z_d_phase": "int32",
    "z_d_total": "int32",
}

in_special_area_dict = {
    "in_bomb_zone": "bool",
    "in_buy_zone": "bool",
    "in_no_defuse_area": "bool",
}

is_movement_dict = {
    "is_ducking": "bool",
    "is_falling": "bool",
    "is_jumping": "bool",
    "is_walking": "bool",
}

is_weapon_dict = {
    "is_burst_mode": "bool",
    "is_reloading": "bool",
    "is_scoping": "bool",
    "is_silencing": "bool",
}

weapon_buyer_dict = {
    "weapon_bought_by_opponent": "bool",
    "weapon_bought_by_self": "bool",
    "weapon_bought_by_teammate": "bool",
}

weapon_type_dict = {
    "weapon_type_c4": "bool",
    "weapon_type_grenade": "bool",
    "weapon_type_knife": "bool",
    "weapon_type_lmg": "bool",
    "weapon_type_nnnnn": "bool",
    "weapon_type_pistol": "bool",
    "weapon_type_rifle": "bool",
    "weapon_type_shotgun": "bool",
    "weapon_type_smg": "bool",
    "weapon_type_sniper": "bool",
}

weapon_name_dict = {
    "weapon_name_ak-47": "bool",
    "weapon_name_aug": "bool",
    "weapon_name_awp": "bool",
    "weapon_name_blaze-grenade": "bool",
    "weapon_name_c4": "bool",
    "weapon_name_cz75-auto": "bool",
    "weapon_name_decoy-grenade": "bool",
    "weapon_name_deserteagle": "bool",
    "weapon_name_dualberettas": "bool",
    "weapon_name_famas": "bool",
    "weapon_name_five-seven": "bool",
    "weapon_name_flashbang-grenade": "bool",
    "weapon_name_frag-grenade": "bool",
    "weapon_name_g3sg1": "bool",
    "weapon_name_galilar": "bool",
    "weapon_name_glock-18": "bool",
    "weapon_name_knife": "bool",
    "weapon_name_m249": "bool",
    "weapon_name_m4a1-s": "bool",
    "weapon_name_m4a4": "bool",
    "weapon_name_mac-10": "bool",
    "weapon_name_mag-7": "bool",
    "weapon_name_mp5-sd": "bool",
    "weapon_name_mp7": "bool",
    "weapon_name_mp9": "bool",
    "weapon_name_negev": "bool",
    "weapon_name_nnnnn": "bool",
    "weapon_name_nova": "bool",
    "weapon_name_p2000": "bool",
    "weapon_name_p250": "bool",
    "weapon_name_p90": "bool",
    "weapon_name_pp-bizon": "bool",
    "weapon_name_r8revolver": "bool",
    "weapon_name_scar-20": "bool",
    "weapon_name_sg553": "bool",
    "weapon_name_ssg08": "bool",
    "weapon_name_sawed-off": "bool",
    "weapon_name_smoke-grenade": "bool",
    "weapon_name_tec-9": "bool",
    "weapon_name_ump-45": "bool",
    "weapon_name_usp-s": "bool",
    "weapon_name_xm1014": "bool",
    "weapon_name_zeusx27": "bool",
}

weapon_ammo_dict = {
    "weapon_ammo": "uint8",
    "weapon_ammo_fired": "uint8",
    "weapon_ammo_remaining": "uint8",
}

weapon_modifier_dict = {
    "accuracy_penalty": "float32",
    "zoom_level": "float32",
}

duck_dict = {"duck_amount": "float32", }

observability_dict = {
    "heard_footsteps": "bool",
    "heard_gunfire": "bool",
    "seen": "bool",
}


agent_state_dict = {
    "is_alive": is_alive_dict,
    "debuffs": debuffs_dict,
    "health": health_dict,
    "money": money_dict,
    "kinematics": [xyz_kin_dict, pvlane_kin_dict][0],
    "lane": [three_lane_dict, relevance_lane_dict][0],
    "defuser": defuser_dict,
    "movement_type": movement_type_dict,
    "alive": alive_dict,
    "assists": assists_dict,
    "cash_spent": cash_spent_dict,
    "damage": damage_dict,
    "deaths": deaths_dict,
    "flashes": flashes_dict,
    "headshot_kills": headshot_kills_dict,
    "headshot_ratio": headshot_ratio_dict,
    "kd_ratio": kd_ratio_dict,
    "kd_spread": kd_spread_dict,
    "kills": kills_dict,
    "objectives": objectives_dict,
    "starting_cash": starting_cash_dict,
    "utility_damage": utility_damage_dict,
    "utility_ratio": utility_ratio_dict,
    "x_delta": x_d_dict,
    "y_delta": y_d_dict,
    "z_delta": z_d_dict,
    "in_special_area": in_special_area_dict,
    "is_movement": is_movement_dict,
    "is_weapon": is_weapon_dict,
    "weapon_buyer": weapon_buyer_dict,
    "weapon_type": weapon_type_dict,
    "weapon_name": weapon_name_dict,
    "weapon_ammo": weapon_ammo_dict,
    "weapon_modifier": weapon_modifier_dict,
    "duck": duck_dict,
    "observability": observability_dict,
}


score_dict = {"score_difference", }

follows_timeouts_dict = {
    "follows_ct_timeout": "bool",
    "follows_technical_timeout": "bool",
    "follows_t_timeout": "bool",
}

num_grenades_across_map_dict = {
    "grenades_map": {
        "num_blazes": "uint8",
        "num_decoys": "uint8",
        "num_flashbangs": "uint8",
        "num_frags": "uint8",
        "num_smokes": "uint8",
    },
    "grenades_a": {
        "num_blazes_a": "uint8",
        "num_decoys_a": "uint8",
        "num_flashbangs_a": "uint8",
        "num_frags_a": "uint8",
        "num_smokes_a": "uint8",
    },
    "grenades_b": {
        "num_blazes_b": "uint8",
        "num_decoys_b": "uint8",
        "num_flashbangs_b": "uint8",
        "num_frags_b": "uint8",
        "num_smokes_b": "uint8",
    },
    "grenades_m": {
        "num_blazes_m": "uint8",
        "num_decoys_m": "uint8",
        "num_flashbangs_m": "uint8",
        "num_frags_m": "uint8",
        "num_smokes_m": "uint8",
    },
}

num_grenades_types_dict = {
    "blazes": {
        "num_blazes": "uint8",
        "num_blazes_a": "uint8",
        "num_blazes_b": "uint8",
        "num_blazes_m": "uint8",
    },
    "decoys": {
        "num_decoys": "uint8",
        "num_decoys_a": "uint8",
        "num_decoys_b": "uint8",
        "num_decoys_m": "uint8",
    },
    "flashbangs": {
        "num_flashbangs": "uint8",
        "num_flashbangs_a": "uint8",
        "num_flashbangs_b": "uint8",
        "num_flashbangs_m": "uint8",
    },
    "frags": {
        "num_frags": "uint8",
        "num_frags_a": "uint8",
        "num_frags_b": "uint8",
        "num_frags_m": "uint8",
    },
    "smokes": {
        "num_smokes": "uint8",
        "num_smokes_a": "uint8",
        "num_smokes_b": "uint8",
        "num_smokes_m": "uint8",
    },
}

num_players_dict = {
    "num_players_ct": "uint8",
    "num_players_t": "uint8",
    "num_players_total": "uint8",
}

bomb_dict = {
    "is_bomb_dropped": "bool",
    "is_bomb_planted": "bool",
}

map_name_dict = {
    "map_name_ancient": "bool",
    "map_name_anubis": "bool",
    "map_name_dust2": "bool",
    "map_name_inferno": "bool",
    "map_name_mirage": "bool",
    "map_name_nuke": "bool",
    "map_name_overpass": "bool",
    "map_name_train": "bool",
    "map_name_vertigo": "bool",
}

map_num_dict = {"map_num": "uint8", }


global_state_dict = {
    "score": score_dict,
    "follows_timeouts": follows_timeouts_dict,
    "num_grenades": [num_grenades_across_map_dict, num_grenades_types_dict][0],
    "num_players": num_players_dict,
    "bomb_status": bomb_dict,
    "map_name": map_name_dict,
    "map_num": map_num_dict,
}


reward_ct_dict = {"reward_ct": "float32", }

reward_t_dict = {"reward_t": "float32", }


reward_dict = {
    "reward_ct": reward_ct_dict,
    "reward_t": reward_t_dict,
}


cols_fillna = {
    "bool": False,
    "int8": np.int8(0),
    "uint8": np.uint8(0),
    "int32": np.int32(0),
    "float32": np.float32(0.0),
    "string": "nnnnn"
}


round_start_pad, round_end_pad = {}, {}
for c, t in all_cols.items():
    round_start_pad[c] = cols_fillna[t]
    round_end_pad[c] = cols_fillna[t]
round_start_pad["start_of_round"] = True
round_start_pad["type"] = initial_type_str
round_end_pad["end_of_round"] = True
round_end_pad["type"] = terminal_type_str


action_colnames = [
    c for c in list({k: v for k, v in recurse_dict(action_dict)}.keys())
]


dataset_colnames = [
    "observations", "n_observations",
    "actions", "rewards", "dones",
    "informations", "n_informations",
]


"""FUNCTIONS"""


def get_parent_directory(fp: str):
    return os.path.basename(os.path.dirname(fp))


def download_demo_archives(
        event_url: str="", count: int=0, sleep_time: float=pause_time
) -> str:

    if len(event_url.split("=")) == 2:
        tournament_str = event_url.split("=")[1]
    else:  tournament_str = "tmp"
    tournament_dir = os.path.join(data_dir, tournament_str)
    if os.path.exists(tournament_dir):
        [[os.remove(os.path.join(tournament_dir, filename))
        for filename in filenames]
        for _, _, filenames in os.walk(tournament_dir)]
    else:  os.makedirs(tournament_dir)

    header = {
        "User-Agent": (UserAgent()).random,
        "Access-Control-Allow-Origin": "*",
    }
    response = requests.get(event_url, headers=header)
    soup = BeautifulSoup(response.text, features="html.parser")
    match_soup = soup.find_all("div", {"class": "result-con"})
    match_pages = []
    for match in match_soup:  match_pages.append(match.find("a")["href"])
    match_pages = reversed(match_pages)
    match_pages = [(hltv_url_str + mch_pg) for mch_pg in match_pages]

    if count <= 0:  mp_count = None
    else:  mp_count = count + 0
    for mch_pg in match_pages[:mp_count]:
        exp_sleep_time = np.random.exponential(sleep_time)
        sleep(exp_sleep_time)
        header = {
            "User-Agent": (UserAgent()).random,
            "Access-Control-Allow-Origin": "*",
        }
        response = requests.get(mch_pg, headers=header)
        soup = BeautifulSoup(response.text, features="html.parser")
        try:
            match_a = soup.find("a", {"class": "stream-box"})
            end_url = match_a["data-demo-link"]
            header = {
                "User-Agent": (UserAgent()).random,
                "Access-Control-Allow-Origin": "*",
            }
            url = hltv_url_str + end_url
            demo_path = \
                os.path.join(tournament_dir, f"{url.split('/')[-1]}.rar")
            with requests.get(url, headers=header, stream=True) as r:
                if os.path.exists(demo_path):  os.remove(demo_path)
                with open(demo_path, "wb") as f:
                    r.raw.decode_content = True
                    shutil.copyfileobj(r.raw, f)
        except:
            print("Demo file cannot be downloaded.")
            return ""

    return tournament_dir


def extract_demo_files(tournament_dir: str="") -> str:

    try:
        [[extract_archive(
            os.path.join(tournament_dir, filename),
            outdir=os.path.join(tournament_dir, filename[:-4]),
            program=os.path.join(assets_dir, "7zG.exe"))
        for filename in filenames]
        for _, _, filenames in os.walk(tournament_dir)]
    except:
        print("Archive cannot be extracted.")
        return ""

    return os.path.basename(tournament_dir)


def retrieve_demo_paths(
        tournaments: list[str]=[],
        teams: list[str]=[], matches: list[tuple[str]]=[]
) -> list[list[str]]:

    tournament_names = \
        [f for f in os.listdir(data_dir) if not f.endswith(".rar")]
    if len(tournaments):
        tournament_names = [f for f in tournament_names if f in tournaments]
    tournament_names = [os.path.join(data_dir, tn) for tn in tournament_names]

    match_names, match_team_names, match_match_names = [], [], []
    for tn in tournament_names:
        if os.path.isdir(tn):
            match_tn_names = [os.path.join(tn, f) for f in os.listdir(tn)
                if not f.endswith(".rar")]
            match_names += match_tn_names
            if len(teams):
                for team in teams:
                    match_team_names += [mn for mn in match_tn_names
                        if team in os.path.basename(mn)]
            if len(matches):
                for match in matches:
                    match_match_names += [mn for mn in match_tn_names
                        if match[0] in os.path.basename(mn)
                        and match[1] in os.path.basename(mn)
                    ]

    if len(matches):  demo_names = match_match_names[:]
    if len(teams) and not len(matches):  demo_names = match_team_names[:]
    if not len(teams) and not len(matches):  demo_names = match_names[:]

    # demos = []
    # for dn in sorted(demo_names):
    #     dn_demos = [demo for demo in os.listdir(dn) if demo.endswith(".dem")]
    #     dn_demos_f = \
    #         sorted(dn_demos, key=lambda s: re.search("-m[0-9]-", s).group())
    #     demos.append(dn_demos_f)

    demos = sorted(demo_names, key=lambda s: re.search("-m[0-9]-", s).group())

    return demos


def find_grenade_pairs_fw(begin: list, expires: list[list]) -> tuple[int, int]:
    if begin == []:  return (-1, -1)
    if len(expires) == 0:  return (-1, -1)
    elif len(expires) == 1:  return (begin[0], expires[0][0])
    else:
        pairs = [(begin, expire, expire[-1] - begin[-1]) for expire in expires
            if begin[-1] <= expire[-1] and \
            begin[2] == expire[2] and begin[3] == expire[3] and \
            expire[4] == begin[4] and \
            expire[5] == begin[5] and \
            expire[6] == begin[6] and \
            begin[0] != expire[0]]
        cnds = sorted(pairs, key=lambda t: t[-1])
        if len(cnds) == 0:  return (-1, -1)
        else:  cnd = cnds.pop(0)
        return (begin[0], cnd[1][0])


def find_grenade_pairs_bw(begins: list[list], expire: list) -> tuple[int, int]:
    if expire == []:  return (-1, -1)
    if len(begins) == 0:  return (-1, -1)
    elif len(begins) == 1:  return (begins[0][0], expire[0])
    else:
        pairs = [(begin, expire, expire[-1] - begin[-1]) for begin in begins
            if expire[-1] >= begin[-1] and \
            expire[2] == begin[2] and expire[3] == begin[3] and \
            begin[4] == expire[4] and \
            begin[5] == expire[5] and \
            begin[6] == expire[6] and \
            expire[0] != begin[0]]
        cnds = sorted(pairs, key=lambda t: t[-1])
        if len(cnds) == 0:  return (-1, -1)
        else:  cnd = cnds.pop(0)
        return (cnd[0][0], expire[0])


def deduplicate_round_markers(
    l: list[int], start: bool=False, end: bool=False
) -> list[int]:
    if start and end:  start, end = False, False
    if not start and not end:  return l
    to_remove = []
    for i in range(len(l) - 1):
        if l[i + 1] - l[i] <= tick_count * minimum_round_duration:
            if start:  to_remove.append(i)
            if end:  to_remove.append(i + 1)
    for t in sorted(to_remove, reverse=True):  l.pop(t)
    return l


def get_round_phase(r):
    if r <= 0:  return 0
    elif r < 25:
        if (r - 0) % 12 == 0:  return 12
        else:  return (r - 0) % 12
    else:
        if (r - 24) % 3 == 0:  return 3
        else:  return (r - 24) % 3


def pad_dfs(df: DataFrame, players: list[str]) -> DataFrame:
    if df.shape[0] == players_count:  return df.copy()
    if df.shape[0] < players_count:
        df_tick = df.iloc[0].tick
        df_players = list(df.player_steamid)
        new_rows = DataFrame({
            "tick": df_tick,
            "player_steamid": [p for p in players if p not in df_players],
            "is_alive": False,
        })
        new_df = concat([df, new_rows], axis=0)
        return new_df.copy()


def parse_demo(
    demo: str, parse_first_half: bool=False, parse_second_half: bool=False,
) -> tuple[DataFrame, str]:

    sleep(pause_time)
    try:  parser = DemoParser(demo)
    except:
        print("Demo file cannot be parsed.")
        return DataFrame()

    header = parser.parse_header()

    events = parser.parse_events(["all"])
    for e in events:  e[1]["type"] = e[0]
    event = concat([e[1] for e in events])
    event = event.reset_index(drop=True)
    event["tick"] = event["tick"].fillna(-1).astype(np.int32)
    event["name"] = \
        event["name"].fillna(cols_fillna[str_str]).astype(str_str)
    event["steamid"] = \
        event["steamid"].fillna(cols_fillna[str_str]).astype(str_str)
    event["user_name"] = \
        event["user_name"].fillna(cols_fillna[str_str]).astype(str_str)
    event["user_steamid"] = \
        event["user_steamid"].fillna(cols_fillna[str_str]).astype(str_str)
    event = event.copy()

    event["attacker_steamid"] = event["attacker_steamid"]\
        .fillna(cols_fillna[str_str]).astype(str_str)
    event["assister_steamid"] = event["assister_steamid"]\
        .fillna(cols_fillna[str_str]).astype(str_str)
    event["defender_steamid"] = cols_fillna[str_str]
    event.loc[event["attacker_steamid"] != cols_fillna[str_str],
        "defender_steamid"] = \
        event.loc[event["attacker_steamid"] != cols_fillna[str_str],
        "user_steamid"]

    event.loc[event.type == old_round_start_str, "type"] = round_start_str
    event["type"] = event["type"].str.replace("begin_", "commence_")
    event["type"] = event["type"].str.replace("grenade", "")
    event["type"] = event["type"].str.replace("he_", "frag_")
    event["type"] = event["type"].str.replace("inferno_", "blaze_")
    event["type"] = event["type"].str.replace("_started", "_begin")
    event["type"] = event["type"].str.replace("_startburn", "_begin")
    event["type"] = event["type"].str.replace("decoy_detonate", "decoy_expire")
    event["type"] = event["type"].str.replace("flashbang_", "flash_")
    event["type"] = event["type"].str.replace("smoke_detonate", "smoke_begin")
    event["type"] = event["type"].str.replace("_expired", "_expire")
    event["type"] = event["type"].str.replace("c4", "bomb")
    event["type"] = event["type"].str.replace("entity_", "player_")
    event.loc[event.type == old_round_end_str, "type"] = round_end_str

    event[["death", "disconnect"]] = False
    event.loc[event.type == "player_death", "death"] = True
    event.loc[event.type == "player_disconnect", "disconnect"] = True

    event["grenade_tmp"] = False
    event.loc[
        (event.type.str.endswith("_begin")) | \
        (event.type.str.endswith("_expire")) | \
        (event.type.str.endswith("_detonate")),
        "grenade_tmp"] = True
    event["grenade_type_tmp"] = cols_fillna[str_str]
    event.loc[event.grenade_tmp, "grenade_type_tmp"] = \
        event.loc[event.grenade_tmp, "type"].str.split("_").str[0]

    event = event.sort_values(by=["tick", "type", "user_steamid"])
    event["tmp_event_idx"] = list(range(1, event.shape[0] + 1))
    event = event.set_index("tmp_event_idx", drop=False)

    event_nade_begins = event.loc[
        (event.user_steamid != cols_fillna[str_str]) &
        (event.type.str.endswith("_begin")),
        ["tmp_event_idx", "type", "user_steamid", "grenade_type_tmp",
        "x", "y", "z", "tick"]].values.tolist()
    event_nade_expires = event.loc[
        (event.user_steamid != cols_fillna[str_str]) &
        (event.type.str.endswith("_expire")),
        ["tmp_event_idx", "type", "user_steamid", "grenade_type_tmp",
        "x", "y", "z", "tick"]].values.tolist()

    pairs = \
        [find_grenade_pairs_bw(event_nade_begins, ene)\
        for ene in event_nade_expires]
    irows = []
    for t in pairs:
        brow = event.iloc[t[0] - 1]
        erow = event.iloc[t[1] - 1]
        for k in range(brow.tick + 1, erow.tick - 1):
            irow = brow.copy()
            irow["tick"] = k
            irows.append(irow)

    nade_df = DataFrame(irows).reset_index(drop=True)
    nade_df["type"] = nade_df["type"].str.replace("_begin", "_detonate")
    nade_df["grenade"] = True
    nade_df["grenade_type"] = nade_df["type"].str.split("_").str[0]

    event["grenade"] = False
    event["grenade_type"] = cols_fillna[str_str]

    event = concat([event, nade_df], axis=0)
    event = event.reset_index(drop=True)
    event = event.copy()

    event = event.drop(columns=
        ["tmp_event_idx", "grenade_tmp", "grenade_type_tmp"])

    event = event.drop(event.loc[event.type.str.endswith("_begin"),].index)
    event = event.drop(event.loc[event.type.str.endswith("_expire"),].index)
    event = event.copy()

    mch_st_tick = event.loc[event.type == round_start_str,].tick.min()
    mch_ed_tick = event.loc[event.type == round_end_str,].tick.max()

    burns_df = event.loc[
        ((event.type == "player_hurt")) &
        ((event.weapon == "molotov") | (event.weapon == "inferno")),
        ["tick", "user_steamid"]]
    burns_df["user_steamid"] = burns_df["user_steamid"].astype(str)
    burns_df = burns_df.sort_values(by="tick")
    burns = sorted(
        list(set([tuple(b) for b in burns_df.values.tolist()])),
        key=lambda k: k[0]
    )

    event = event.drop(
        event.loc[(event.tick == -1) | \
        (event.type == cols_fillna[str_str]),].index)
    event = event.drop(event.loc[(event.type != round_start_str) &
        (event.tick <= mch_st_tick),].index)
    event = event.drop(event.loc[(event.type != round_end_str) &
        (event.tick >= mch_ed_tick),].index)
    event = event.drop(event.loc[event.type == "",].index)
    event = event.drop(event.loc[~event.type.isin(old_event_types),].index)
    for etk, etv in event_rename_types.items():
        event.loc[event.type == etk, "type"] = etv
    event["x_rounded"] = event["x"].round(2)
    event["y_rounded"] = event["y"].round(2)
    event["z_rounded"] = event["z"].round(2)
    event = event.drop_duplicates(
        subset=[
            "tick", "type", "user_steamid",
            "x_rounded", "y_rounded", "z_rounded",
        ],
        keep="first"
    )
    event = event.drop(columns=["x_rounded", "y_rounded", "z_rounded"])
    event = event.sort_values(by=["tick", "type", "user_steamid"])

    event = event.rename(columns={
        "dmg_armor": "damage_armor",
        "dmg_health": "damage_health",
        "x": "x_loc",
        "y": "y_loc",
        "z": "z_loc",
    })

    event[["not_penetrated", "single_penetrated", "double_penetrated"]] = False
    event.loc[event.penetrated == 0.0, "not_penetrated"] = True
    event.loc[event.penetrated == 1.0, "single_penetrated"] = True
    event.loc[event.penetrated == 2.0, "double_penetrated"] = True

    event["weapon"] = event["weapon"].str.replace("weapon_", "")

    event[[
        "hitgroup_head", "hitgroup_chest", "hitgroup_stomach",
        "hitgroup_leftarm", "hitgroup_rightarm",
        "hitgroup_leftleg", "hitgroup_rightleg",
        "hitgroup_generic", "hitgroup_gear"
    ]] = False
    event.loc[event.hitgroup == 0, "hitgroup_generic"] = True
    event.loc[event.hitgroup == 1, "hitgroup_head"] = True
    event.loc[event.hitgroup == 2, "hitgroup_chest"] = True
    event.loc[event.hitgroup == 3, "hitgroup_stomach"] = True
    event.loc[event.hitgroup == 4, "hitgroup_leftarm"] = True
    event.loc[event.hitgroup == 5, "hitgroup_rightarm"] = True
    event.loc[event.hitgroup == 6, "hitgroup_leftleg"] = True
    event.loc[event.hitgroup == 7, "hitgroup_rightleg"] = True
    event.loc[event.hitgroup == 10, "hitgroup_gear"] = True

    df = parser.parse_ticks(
        wanted_props=tick_attrs + [a.upper() for a in action_buttons],
        ticks=[]
    )
    df = df.reset_index(drop=True)
    df["tick"] = df["tick"].fillna(-1).astype(np.int32)
    df["player_name"] = \
        df["player_name"].fillna(cols_fillna[str_str]).astype(str_str)
    df["player_steamid"] = \
        df["player_steamid"].fillna(cols_fillna[str_str]).astype(str_str)
    df = df.copy()

    df = df.drop(df.loc[(df.tick == -1) | \
        (df.player_steamid == cols_fillna[str_str]),].index)
    df = df.drop(df.loc[df.player_name == "device",].index)
    df = df.drop(columns=["name", "steamid"])
    df = df.drop_duplicates(subset=["tick", "player_steamid"], keep="first")
    df = df.sort_values(by=["tick", "player_steamid"])

    df = df.rename(columns={
        "last_place_name": "place",
        "num_terrorist_timeouts": "num_t_timeouts",
        "total_rounds_played": "round_total",
        "rounds_played_this_phase": "round_phase",
        "health": "health_value",
        "balance": "cash_value",
        "start_balance": "starting_cash",
        "current_equip_value": "equipment_value",
        "ducking": "is_ducking",
        "is_silencer_on": "is_silencing",
        "is_scoped": "is_scoping",
        "is_in_reload": "is_reloading",
        "X": "x_p",
        "Y": "y_p",
        "Z": "z_p",
        "velocity_X": "x_v",
        "velocity_Y": "y_v",
        "velocity_Z": "z_v",
        "shots_fired": "weapon_ammo_fired",
        "total_ammo_left": "weapon_ammo_remaining",
        "enemies_flashed_total": "flashes_total",
        "objective_total": "objectives_total",
        "old_jump_pressed": "is_jumping",
        "zoom_lvl": "zoom_level",
        "FIRE": "fire",
        "RELOAD": "reload",
        "USE": "use",
        "FORWARD": "forward",
        "LEFT": "left",
        "RIGHT": "right",
        "BACK": "back",
        "team_clan_name": "player_team",
        "approximate_spotted_by": "spotted_by",
    })

    df["player_team"] = df["player_team"].str.replace(" ", "-")
    df["player_team"] = df["player_team"].str.upper()

    df["armor_value"] = df["armor_value"] / 100.0
    df["health_value"] = df["health_value"] / 100.0

    df["round_total"] = df.round_total.add(1)
    df["round_phase"] = df.round_phase.add(1)

    rounds = \
        sorted([r for r in list(dict.fromkeys(df.round_total).keys())
        if r is not None])
    teams = \
        sorted([t for t in list(dict.fromkeys(df.player_team).keys())
        if t is not None])
    players = \
        sorted([p for p in list(dict.fromkeys(df.player_steamid).keys())
        if p is not None])
    if len(teams) != 2:  return DataFrame()
    if len(players) != players_count:  return DataFrame()

    event["map_name"] = cols_fillna[str_str]
    event["map_num"] = 0
    event["server_name"] = cols_fillna[str_str]
    event["tournament"] = cols_fillna[str_str]
    event["match"] = cols_fillna[str_str]

    ticks = []
    warmup_ticks = df.loc[df.is_warmup_period,].tick.tolist()
    freeze_ticks = df.loc[df.is_freeze_period,].tick.tolist()
    rd_st_ticks = event.loc[
        (event.type == round_start_str) & (event.tick != mch_st_tick) &
        (~event.tick.isin(warmup_ticks)) & (~event.tick.isin(freeze_ticks)),]\
        .tick.tolist()
    rd_ed_ticks = event.loc[
        (event.type == round_end_str) & (event.tick != mch_ed_tick) &
        (~event.tick.isin(warmup_ticks)) & (~event.tick.isin(freeze_ticks)),]\
        .tick.tolist()
    rd_st_ticks = sorted(list(set(rd_st_ticks)))
    rd_ed_ticks = sorted(list(set(rd_ed_ticks)))
    rd_st_ticks = sorted(deduplicate_round_markers(rd_st_ticks, start=True))
    rd_ed_ticks = sorted(deduplicate_round_markers(rd_ed_ticks, end=True))
    rd_st_ticks = [mch_st_tick] + rd_st_ticks
    rd_ed_ticks = rd_ed_ticks + [mch_ed_tick]
    if len(rd_st_ticks) != len(rd_ed_ticks):
        return DataFrame()

    rounds = rounds[:((len(rd_st_ticks) + len(rd_ed_ticks)) // 2)]
    for rd in range(len(rounds)):
        ticks_rd = [t for t in range(rd_st_ticks[rd], rd_ed_ticks[rd] + 1)]
        ticks += ticks_rd
    ticks = sorted(list(set(ticks)))

    df.loc[df.tick.isin(rd_ed_ticks), "round_total"] = \
        df.loc[df.tick.isin(rd_ed_ticks), "round_total"].sub(1)
    df.loc[df.tick.isin(rd_ed_ticks), "round_phase"] = \
        df.loc[df.tick.isin(rd_ed_ticks), "round_phase"].sub(1)
    df.loc[df.round_phase == 0, "round_phase"] = np.nan
    df["round_phase"] = df["round_phase"].ffill()

    t_timeout_rounds = df.loc[df.is_terrorist_timeout,].round_total.tolist()
    ct_timeout_rounds = df.loc[df.is_ct_timeout,].round_total.tolist()
    neutral_timeout_rounds = df.loc[df.is_technical_timeout,].round_total\
        .tolist()
    t_timeout_rounds = sorted(list(dict.fromkeys(t_timeout_rounds).keys()))
    ct_timeout_rounds = sorted(list(dict.fromkeys(ct_timeout_rounds).keys()))
    neutral_timeout_rounds = \
        sorted(list(dict.fromkeys(neutral_timeout_rounds).keys()))
    df[["follows_t_timeout", "follows_ct_timeout"]] = False
    df.loc[df.round_total.isin(t_timeout_rounds), "follows_t_timeout"] = True
    df.loc[df.round_total.isin(ct_timeout_rounds), "follows_ct_timeout"] = True
    df["follows_technical_timeout"] = False
    df.loc[df.round_total.isin(neutral_timeout_rounds),
        "follows_technical_timeout"] = True

    df = df.drop(df.loc[~df.tick.isin(ticks),].index)

    event = event.drop(event.loc[~event.tick.isin(ticks),].index)
    event = event.drop(event.loc[(event.type != round_start_str) &
        (event.tick <= ticks[0]),].index)
    event = event.drop(event.loc[(event.type != round_end_str) &
        (event.tick >= ticks[-1]),].index)
    event = event.drop(event.loc[~event.tick.isin(list(df.tick)),].index)

    df[[
        "start_of_match", "start_of_phase", "start_of_round",
        "end_of_round", "end_of_phase", "end_of_match"
    ]] = False
    df.loc[
        (df.round_total == df.round_total.min()) & (df.tick.isin(rd_st_ticks)),
        "start_of_match"] = True
    df.loc[
        (df.round_total.isin([1, 13, 25, 28])) & (df.tick.isin(rd_st_ticks)),
        "start_of_phase"] = True
    df.loc[df.tick.isin(rd_st_ticks), "start_of_round"] = True
    df.loc[df.tick.isin(rd_ed_ticks), "end_of_round"] = True
    df.loc[
        (df.round_total == df.round_total.max()) & (df.tick.isin(rd_ed_ticks)),
        "end_of_phase"] = True
    df.loc[
        (df.round_total.isin([12, 24, 27, 30])) & (df.tick.isin(rd_ed_ticks)),
        "end_of_match"] = True

    df = df.drop(df.loc[df.round_total >= 25,].index)
    df = df.copy()

    df[[
        "first_half", "second_half",
    ]] = False
    df.loc[(df.round_total >= 1) & (df.round_total < 13),
        "first_half"] = True
    df.loc[(df.round_total >= 13) & (df.round_total < 25),
        "second_half"] = True

    df[[c for c in df.columns if "time" in c]] = \
        df[[c for c in df.columns if "time" in c]].replace(-1, 0)

    df["time_since_spawn"] = df.game_time - df.spawn_time
    df["time_since_death"] = df.game_time - df.death_time

    df.loc[~df.is_alive, ["x_p", "y_p", "z_p"]] = np.nan

    df["team_num"] = df["team_num"].replace(ct_str, 3)
    df["team_num"] = df["team_num"].replace("counter-terrorist", 3)
    df["team_num"] = df["team_num"].replace("Counter-terrorist", 3)
    df["team_num"] = df["team_num"].replace("Counter-Terrorist", 3)
    df["team_num"] = df["team_num"].replace("COUNTER-TERRORIST", 3)
    df["team_num"] = df["team_num"].replace(t_str, 2)
    df["team_num"] = df["team_num"].replace("terrorist", 2)
    df["team_num"] = df["team_num"].replace("Terrorist", 2)
    df["team_num"] = df["team_num"].replace("TERRORIST", 2)
    df["team_num"] = df["team_num"]\
        .fillna(cols_fillna["uint8"]).astype(np.uint8)

    df[["is_t", "is_ct"]] = False
    df.loc[df.team_num == 2, "is_t"] = True
    df.loc[df.team_num == 3, "is_ct"] = True

    df["is_alive_int"] = df.is_alive.astype(np.uint8)
    df["alive_total"] = \
        df[["player_steamid", "round_total", "is_alive_int"]]\
        .groupby(["player_steamid", "round_total"]).transform("sum")
    df.loc[df.start_of_match, "alive_total"] = 0
    df.loc[~df.start_of_round, "alive_total"] = 0
    df["alive_total"] = df.alive_total.floordiv(tick_count)
    df["alive_total"] = df[["player_steamid", "alive_total"]]\
        .groupby("player_steamid").transform("cumsum")

    df.loc[~df.start_of_round, "starting_cash"] = 0.0
    df["starting_cash_total"] = df[["player_steamid", "starting_cash"]]\
        .groupby("player_steamid").transform("cumsum")

    df.loc[~df.start_of_round, "cash_spent_this_round"] = 0.0
    df["cash_spent_total"] = \
        df[["player_steamid", "cash_spent_this_round"]]\
        .groupby("player_steamid").transform("cumsum")

    df[["x_d", "y_d", "z_d"]] = 0.0
    for player in players:
        if player != cols_fillna[str_str]:
            df.loc[df.player_steamid == player, "x_d"] = \
                df.loc[df.player_steamid == player, "x_p"] - \
                df.loc[df.player_steamid == player, "x_p"].shift(1)
            df.loc[df.player_steamid == player, "y_d"] = \
                df.loc[df.player_steamid == player, "y_p"] - \
                df.loc[df.player_steamid == player, "y_p"].shift(1)
            df.loc[df.player_steamid == player, "z_d"] = \
                df.loc[df.player_steamid == player, "z_p"] - \
                df.loc[df.player_steamid == player, "z_p"].shift(1)

    df["x_d"] = df["x_d"].abs()
    df["y_d"] = df["y_d"].abs()
    df["z_d"] = df["z_d"].abs()
    df.loc[df.x_d < epsilon, "x_d"] = 0.0
    df.loc[df.y_d < epsilon, "y_d"] = 0.0
    df.loc[df.z_d < epsilon, "z_d"] = 0.0
    df.loc[df.start_of_round, ["x_d", "y_d", "z_d"]] = 0.0
    df["x_d_total"] = \
        df[["player_steamid", "round_total", "x_d"]]\
        .groupby(["player_steamid", "round_total"])\
        .transform("sum")
    df["y_d_total"] = \
        df[["player_steamid", "round_total", "y_d"]]\
        .groupby(["player_steamid", "round_total"])\
        .transform("sum")
    df["z_d_total"] = \
        df[["player_steamid", "round_total", "z_d"]]\
        .groupby(["player_steamid", "round_total"])\
        .transform("sum")
    df.loc[df.start_of_match, ["x_d", "y_d", "z_d", \
        "x_d_total", "y_d_total", "z_d_total"]] = 0.0
    df.loc[~df.start_of_round,
        ["x_d_total", "y_d_total", "z_d_total"]] = 0.0
    df["x_d_total"] = df[["player_steamid", "x_d_total"]]\
        .groupby("player_steamid").transform("cumsum")
    df["y_d_total"] = df[["player_steamid", "y_d_total"]]\
        .groupby("player_steamid").transform("cumsum")
    df["z_d_total"] = df[["player_steamid", "z_d_total"]]\
        .groupby("player_steamid").transform("cumsum")

    df["is_falling"] = False
    df.loc[df.fall_velo.abs() > epsilon, "is_falling"] = True

    df[["moving_default", "moving_ladder"]] = False
    df.loc[df.move_type == 2, "moving_default"] = True
    df.loc[df.move_type == 9, "moving_ladder"] = True

    df[[
        "active_weapon_bought_by_0", "weapon_bought_by_self",
        "weapon_bought_by_teammate", "weapon_bought_by_opponent"
    ]] = False
    df.loc[
        (df.active_weapon_name.notna()) &
        (df.active_weapon_name.str.contains(
            "knife", case=False, na=cols_fillna[str_str], regex=False)) |
        (df.active_weapon_name.str.contains(
            "bayonet", case=False, na=cols_fillna[str_str], regex=False)) |
        (df.active_weapon_name.str.contains(
            "karambit", case=False, na=cols_fillna[str_str], regex=False)) |
        (df.active_weapon_name.str.contains(
            "dagger", case=False, na=cols_fillna[str_str], regex=False)),
        "active_weapon_bought_by_0"] = True
    df.loc[
        (df.active_weapon_name.notna()) &
        (df.player_steamid == df.active_weapon_original_owner.astype(str_str)),
        "active_weapon_bought_by_0"] = True
    df.loc[(df.active_weapon_name.notna()) & (df.active_weapon_bought_by_0),
        "weapon_bought_by_self"] = True
    df.loc[
        (df.active_weapon_name.notna()) & (df.orig_team_number.notna()) &
        (df.team_num == df.orig_team_number) & (~df.active_weapon_bought_by_0),
        "weapon_bought_by_teammate"] = True
    df.loc[
        (df.active_weapon_name.notna()) & (df.orig_team_number.notna()) &
        (df.team_num != df.orig_team_number) & (~df.active_weapon_bought_by_0),
        "weapon_bought_by_opponent"] = True

    df["active_weapon_name"] = \
        df.active_weapon_name.fillna(cols_fillna[str_str])
    df.loc[(df.active_weapon_name.notna()) &
        (df.active_weapon_name.str.contains(
            "knife", case=False, na=cols_fillna[str_str], regex=False)) |
        (df.active_weapon_name.str.contains(
            "bayonet", case=False, na=cols_fillna[str_str], regex=False)) |
        (df.active_weapon_name.str.contains(
            "karambit", case=False, na=cols_fillna[str_str], regex=False)) |
        (df.active_weapon_name.str.contains(
            "dagger", case=False, na=cols_fillna[str_str], regex=False)),
        "active_weapon_name"] = "Knife"

    df["active_weapon_type"] = df.active_weapon_name.map(weapon_types)

    df.loc[df.flash_max_alpha == 255.0, "is_flashed"] = False
    df.loc[df.flash_max_alpha == 0.0, "is_flashed"] = True

    df["is_burned"] = False
    for player in players:
        if player != cols_fillna[str_str]:
            new_burns_p = []
            burns_p = sorted([b[0] for b in burns if b[1] == player])
            if len(burns_p) >= 1:
                for bp in range(len(burns_p) - 1):
                    if (burns_p[bp+1] - burns_p[bp]) < ((tick_count // 5) + 3):
                        burn_range = \
                            list(range(burns_p[bp], burns_p[bp+1]))
                        new_burns_p += [br for br in burn_range]
                    else:
                        new_burns_p += [burns_p[bp]]
                new_burns_p += [burns_p[-1]]
            df.loc[df.tick.isin(new_burns_p) & (df.player_steamid == player),
                "is_burned"] = True

    df = df.copy()
    df = df.join(get_dummies(
        df["active_weapon_name"].str.replace(" ", "").str.lower(),
        prefix="weapon_name"
    ))
    df = df.join(get_dummies(df["active_weapon_type"], prefix="weapon_type"))
    df = df.rename(columns={
        "weapon_name_incendiarygrenade": "blazegrenade0",
        "weapon_name_decoygrenade": "weapon_name_decoy-grenade",
        "weapon_name_highexplosivegrenade": "weapon_name_frag-grenade",
        "weapon_name_flashbang": "weapon_name_flashbang-grenade",
        "weapon_name_smokegrenade": "weapon_name_smoke-grenade",
        "weapon_name_molotov": "blazegrenade1",
    })
    df["weapon_name_blaze-grenade"] = False
    df.loc[
        (df.blazegrenade0.astype(bool)) | (df.blazegrenade1.astype(bool)),
        "weapon_name_blaze-grenade"] = True

    df = df.copy()
    df.loc[df.round_total == 1, total_int_stats] = 0
    df.loc[~df.start_of_round, total_int_stats] = np.nan
    for player in players:
        if player != cols_fillna[str_str]:
            df.loc[(df.player_steamid == player), total_int_stats] = \
                df.loc[(df.player_steamid == player), total_int_stats].ffill()

    df = df.copy()
    df["kd_spread_total"] = \
        df.kills_total.astype(int) - df.deaths_total.astype(int)
    for player in players:
        if player != cols_fillna[str_str]:
            for c in total_int_stats:
                cc = c.replace("total", "phase")
                if df.loc[(df.player_steamid == player) &
                        (df.first_half), c].shape[0] > 0:
                    v1 = df.loc[(df.player_steamid == player) &
                        (df.first_half), c].iloc[0]
                    df.loc[(df.player_steamid == player) &
                        (df.first_half), cc] = \
                        df.loc[(df.player_steamid == player) &
                        (df.first_half), c] - v1
                if df.loc[(df.player_steamid == player) &
                        (df.second_half), c].shape[0] > 0:
                    v2 = df.loc[(df.player_steamid == player) &
                        (df.second_half), c].iloc[0]
                    df.loc[(df.player_steamid == player) &
                        (df.second_half), cc] = \
                        df.loc[(df.player_steamid == player) &
                        (df.second_half), c] - v2
    df["kd_spread_phase"] = \
        df.kills_phase.astype(int) - df.deaths_phase.astype(int)

    df = df.copy()
    df.loc[df.deaths_total == 0, "kd_ratio_total"] = \
        df.loc[df.deaths_total == 0,].kills_total / \
        df.loc[df.deaths_total == 0,].deaths_total.add(1)
    df.loc[df.deaths_total > 0, "kd_ratio_total"] = \
        df.loc[df.deaths_total > 0,].kills_total / \
        df.loc[df.deaths_total > 0,].deaths_total
    df["headshot_ratio_total"] = df.headshot_kills_total / df.kills_total
    df["utility_ratio_total"] = df.utility_damage_total / df.damage_total
    df[["headshot_ratio_total", "utility_ratio_total"]] = \
        df[["headshot_ratio_total", "utility_ratio_total"]]\
        .fillna(cols_fillna["float32"])
    df["alive_total"] = df.alive_total.add(1)

    df.loc[df.deaths_phase == 0, "kd_ratio_phase"] = \
        df.loc[df.deaths_phase == 0,].kills_phase / \
        df.loc[df.deaths_phase == 0,].deaths_phase.add(1)
    df.loc[df.deaths_phase > 0, "kd_ratio_phase"] = \
        df.loc[df.deaths_phase > 0,].kills_phase / \
        df.loc[df.deaths_phase > 0,].deaths_phase
    df["headshot_ratio_phase"] = df.headshot_kills_phase / df.kills_phase
    df["utility_ratio_phase"] = df.utility_damage_phase / df.damage_phase
    df[["headshot_ratio_phase", "utility_ratio_phase"]] = \
        df[["headshot_ratio_phase", "utility_ratio_phase"]]\
        .fillna(cols_fillna["float32"])
    df["alive_phase"] = df.alive_phase.add(1)

    df = df.copy()
    for c in total_int_stats + total_ratio_stats:
        df[c.replace("total", "per_round_total")] = \
            df[c] / df["round_total"]
        df[c.replace("total", "per_second_total")] = \
            df[c] / df["alive_total"]
    for c in phase_int_stats + phase_ratio_stats:
        df[c.replace("phase", "per_round_phase")] = \
            df[c] / df["round_phase"]
        df[c.replace("phase", "per_second_phase")] = \
            df[c] / df["alive_phase"]

    df = df.copy()
    df["alive_per_round_total"] = df.alive_total / df.round_total
    df["alive_per_round_phase"] = df.alive_phase / df.round_phase

    df.loc[df.place == "", "place"] = cols_fillna[str_str]

    df.loc[df.place == "A", "place"] = "BombsiteA"
    df.loc[df.place == "B", "place"] = "BombsiteB"
    df.loc[df.place == "ASite", "place"] = "BombsiteA"
    df.loc[df.place == "BSite", "place"] = "BombsiteB"
    df.loc[df.place == "SiteA", "place"] = "BombsiteA"
    df.loc[df.place == "SiteB", "place"] = "BombsiteB"

    a_x_min = df.loc[df.place == "BombsiteA", "x_p"].min()
    a_y_min = df.loc[df.place == "BombsiteA", "y_p"].min()
    a_z_min = df.loc[df.place == "BombsiteA", "z_p"].min()
    a_x_max = df.loc[df.place == "BombsiteA", "x_p"].max()
    a_y_max = df.loc[df.place == "BombsiteA", "y_p"].max()
    a_z_max = df.loc[df.place == "BombsiteA", "z_p"].max()
    b_x_min = df.loc[df.place == "BombsiteB", "x_p"].min()
    b_y_min = df.loc[df.place == "BombsiteB", "y_p"].min()
    b_z_min = df.loc[df.place == "BombsiteB", "z_p"].min()
    b_x_max = df.loc[df.place == "BombsiteB", "x_p"].max()
    b_y_max = df.loc[df.place == "BombsiteB", "y_p"].max()
    b_z_max = df.loc[df.place == "BombsiteB", "z_p"].max()

    a_x = (a_x_min + a_x_max) / 2
    a_y = (a_y_min + a_y_max) / 2
    a_z = (a_z_min + a_z_max) / 2
    b_x = (b_x_min + b_x_max) / 2
    b_y = (b_y_min + b_y_max) / 2
    b_z = (b_z_min + b_z_max) / 2
    m_x = (a_x + b_x) / 2
    m_y = (a_y + b_y) / 2
    m_z = (a_z + b_z) / 2

    event["a_x_loc_distance"] = event["x_loc"].sub(a_x)
    event["a_y_loc_distance"] = event["y_loc"].sub(a_y)
    event["a_z_loc_distance"] = event["z_loc"].sub(a_z)
    event["m_x_loc_distance"] = event["x_loc"].sub(m_x)
    event["m_y_loc_distance"] = event["y_loc"].sub(m_y)
    event["m_z_loc_distance"] = event["z_loc"].sub(m_z)
    event["b_x_loc_distance"] = event["x_loc"].sub(b_x)
    event["b_y_loc_distance"] = event["y_loc"].sub(b_y)
    event["b_z_loc_distance"] = event["z_loc"].sub(b_z)

    event["lane_a_loc_distance"] = np.sqrt(
        (event["a_x_loc_distance"].pow(2)) +
        (event["a_y_loc_distance"].pow(2)) +
        (event["a_z_loc_distance"].pow(2)) +
    (0 ** 2))
    event["lane_m_loc_distance"] = np.sqrt(
        (event["m_x_loc_distance"].pow(2)) +
        (event["m_y_loc_distance"].pow(2)) +
        (event["m_z_loc_distance"].pow(2)) +
    (0 ** 2))
    event["lane_b_loc_distance"] = np.sqrt(
        (event["b_x_loc_distance"].pow(2)) +
        (event["b_y_loc_distance"].pow(2)) +
        (event["b_z_loc_distance"].pow(2)) +
    (0 ** 2))

    event["min_lane_loc_distance"] = \
        event[[
            "lane_a_loc_distance",
            "lane_m_loc_distance",
            "lane_b_loc_distance",
        ]].min(axis=1)

    event[["lane_a", "lane_m", "lane_b"]] = False
    event.loc[event.min_lane_loc_distance == event.lane_a_loc_distance,
        "lane_a"] = True
    event.loc[event.min_lane_loc_distance == event.lane_m_loc_distance,
        "lane_m"] = True
    event.loc[event.min_lane_loc_distance == event.lane_b_loc_distance,
        "lane_b"] = True

    df["lane_a_x_distance"] = df["x_p"].sub(a_x)
    df["lane_a_y_distance"] = df["y_p"].sub(a_y)
    df["lane_a_z_distance"] = df["z_p"].sub(a_z)
    df["lane_m_x_distance"] = df["x_p"].sub(m_x)
    df["lane_m_y_distance"] = df["y_p"].sub(m_y)
    df["lane_m_z_distance"] = df["z_p"].sub(m_z)
    df["lane_b_x_distance"] = df["x_p"].sub(b_x)
    df["lane_b_y_distance"] = df["y_p"].sub(b_y)
    df["lane_b_z_distance"] = df["z_p"].sub(b_z)

    df["lane_a_distance"] = np.sqrt(
        (df["lane_a_x_distance"].pow(2)) +
        (df["lane_a_y_distance"].pow(2)) +
        (df["lane_a_z_distance"].pow(2)) +
    (0 ** 2))
    df["lane_m_distance"] = np.sqrt(
        (df["lane_m_x_distance"].pow(2)) +
        (df["lane_m_y_distance"].pow(2)) +
        (df["lane_m_z_distance"].pow(2)) +
    (0 ** 2))
    df["lane_b_distance"] = np.sqrt(
        (df["lane_b_x_distance"].pow(2)) +
        (df["lane_b_y_distance"].pow(2)) +
        (df["lane_b_z_distance"].pow(2)) +
    (0 ** 2))

    df["min_lane_distance"] = \
        df[["lane_a_distance", "lane_m_distance", "lane_b_distance"]]\
            .min(axis=1)
    df["max_lane_distance"] = \
        df[["lane_a_distance", "lane_m_distance", "lane_b_distance"]]\
            .max(axis=1)

    df[[
        "lane_a_primary", "lane_a_secondary",
        "lane_m_primary", "lane_m_secondary",
        "lane_b_primary", "lane_b_secondary",
    ]] = False

    df.loc[df.min_lane_distance == df.lane_a_distance, "lane_a_primary"] = \
        True
    df.loc[df.min_lane_distance == df.lane_m_distance, "lane_m_primary"] = \
        True
    df.loc[df.min_lane_distance == df.lane_b_distance, "lane_b_primary"] = \
        True

    df.loc[(df.max_lane_distance == df.lane_m_distance) & (df.lane_a_primary),
        "lane_b_secondary"] = True
    df.loc[(df.max_lane_distance == df.lane_b_distance) & (df.lane_a_primary),
        "lane_m_secondary"] = True
    df.loc[(df.max_lane_distance == df.lane_a_distance) & (df.lane_m_primary), \
        "lane_b_secondary"] = True
    df.loc[(df.max_lane_distance == df.lane_b_distance) & (df.lane_m_primary),
        "lane_a_secondary"] = True
    df.loc[(df.max_lane_distance == df.lane_a_distance) & (df.lane_b_primary),
        "lane_m_secondary"] = True
    df.loc[(df.max_lane_distance == df.lane_m_distance) & (df.lane_b_primary),
        "lane_a_secondary"] = True

    df.loc[~df.end_of_round, ["round_win_status", "round_win_reason"]] = 0

    df[[
        "won_with_ct_defused", "won_with_ct_delayed", "won_with_ct_eliminated",
        "won_with_t_eliminated", "won_with_t_exploded",
    ]] = False
    df.loc[df.round_win_reason == 7, "won_with_ct_defused"] = True
    df.loc[df.round_win_reason == 12, "won_with_ct_delayed"] = True
    df.loc[df.round_win_reason == 8, "won_with_ct_eliminated"] = True
    df.loc[df.round_win_reason == 9, "won_with_t_eliminated"] = True
    df.loc[df.round_win_reason == 1, "won_with_t_exploded"] = True

    df[[
        "score_first_half_ct", "score_first_half_t",
        "score_second_half_ct", "score_second_half_t"
    ]] = 0
    for rd in rounds:
        if rd >= 1 and rd < 13:
            tm_score = "team_score_first_half"
        if rd >= 13 and rd < 25:
            tm_score = "team_score_second_half"
        ct_team_score = \
            df.loc[(df.round_total == rd) & (df.is_ct), tm_score].iloc[0]
        t_team_score = \
            df.loc[(df.round_total == rd) & (df.is_t), tm_score].iloc[0]
        if rd >= 1 and rd < 13:
            df.loc[(df.round_total == rd), "score_first_half_ct"] = \
                ct_team_score
            df.loc[(df.round_total == rd), "score_first_half_t"] = \
                t_team_score
        if rd >= 13 and rd < 25:
            df.loc[(df.round_total == rd), "score_second_half_ct"] = \
                ct_team_score
            df.loc[(df.round_total == rd), "score_second_half_t"] = \
                t_team_score

    # df = df.copy()
    # df = df.groupby("tick")\
    #     .apply(lambda d: pad_dfs(d, players), group_keys=False)

    df = df.copy()
    df_cols = [c for c in df.columns]
    df_keep_cols = []
    for c in df.columns:
        for k in ["situation", "round", "game", "match"]:
            if c in cols["node"][k]:
                t = cols["node"][k][c]
                df[c] = df[c].fillna(cols_fillna[t]).astype(t)
                df_keep_cols.append(c)
            elif c in cols["downsample"][k]:
                t = cols["downsample"][k][c]
                df[c] = df[c].fillna(cols_fillna[t]).astype(t)
                df_keep_cols.append(c)
    df_keep_cols.append("spotted_by")
    df_keep_cols.append("tick")
    df_remove_cols = [c for c in df_cols if c not in df_keep_cols]
    df = df.drop(columns=df_remove_cols)
    df = df.copy()

    event = event.copy()
    event_cols = [c for c in event.columns]
    event_keep_cols = []
    for c in event.columns:
        for k in ["situation", "round", "game", "match"]:
            if c in cols["graph"][k]:
                t = cols["graph"][k][c]
                event[c] = event[c].fillna(cols_fillna[t]).astype(t)
                event_keep_cols.append(c)
            elif c in cols["downsample"][k]:
                t = cols["downsample"][k][c]
                event[c] = event[c].fillna(cols_fillna[t]).astype(t)
                event_keep_cols.append(c)
    event_keep_cols.append("tick")
    event_keep_cols.append("attacker_steamid")
    event_keep_cols.append("assister_steamid")
    event_keep_cols.append("defender_steamid")
    event_remove_cols = [c for c in event_cols if c not in event_keep_cols]
    event = event.drop(columns=event_remove_cols)
    event = event.copy()

    cs = 0
    round_start_ends = {
        rd_st_ticks[t]: 0
        for t in range(len(rd_ed_ticks))
    }
    round_start_pad_dict = {t: round_start_pad.copy() for t in rd_st_ticks}
    for rts, rsp in round_start_pad_dict.items():
        cs += 1
        rsp["tick"] = rts - 1
        rsp["timestep"] = round_start_ends[rts]
        rsp["round_total"] = cs * 1
        rsp["round_phase"] = get_round_phase(cs * 1)
        if cs >= 1 and cs < 13:
            rsp["first_half"] = True
            rsp["second_half"] = False
            rsp["score_first_half_ct"] = \
                df.loc[df.tick == rts, "score_first_half_ct"].iloc[0]
            rsp["score_first_half_t"] = \
                df.loc[df.tick == rts, "score_first_half_t"].iloc[0]
        if cs >= 13 and cs < 25:
            rsp["first_half"] = False
            rsp["second_half"] = True
            rsp["score_second_half_ct"] = \
                df.loc[df.tick == rts, "score_second_half_ct"].iloc[0]
            rsp["score_second_half_t"] = \
                df.loc[df.tick == rts, "score_second_half_t"].iloc[0]
        rsp["spotted_by"] = []
        if cs in [1, 13, 25, 28]:  rsp["start_of_phase"] = True
        else:  rsp["start_of_phase"] = False
        if cs == rounds[0]:  rsp["start_of_match"] = True
        else:  rsp["start_of_match"] = False
        rsp["attacker_steamid"] = cols_fillna[str_str]
        rsp["assister_steamid"] = cols_fillna[str_str]
        rsp["defender_steamid"] = cols_fillna[str_str]
        rsp["is_alive"] = not bnd_is_alive
    round_start_pad_df = DataFrame(list(round_start_pad_dict.values()))
    round_start_pad_df_f = round_start_pad_df.loc[
        np.repeat(round_start_pad_df.index, players_count)]\
        .reset_index(drop=True)

    mgd = df.merge(event, how="left", on="tick", suffixes=("", "_event"))
    if mgd.shape[0] <= players_count * tick_count * minimum_round_duration:
        return DataFrame()

    ce = 0
    round_end_starts = {
        rd_ed_ticks[t]: rd_ed_ticks[t] - rd_st_ticks[t] + 1
        for t in range(len(rd_ed_ticks))
    }
    round_end_pad_dict = {t: round_end_pad.copy() for t in rd_ed_ticks}
    for rte, rep in round_end_pad_dict.items():
        ce += 1
        rep["tick"] = rte + 1
        rep["timestep"] = round_end_starts[rte]
        rep["round_total"] = ce * 1
        rep["round_phase"] = get_round_phase(ce * 1)
        if ce >= 1 and ce < 13:
            rep["first_half"] = True
            rep["second_half"] = False
            rep["score_first_half_ct"] = \
                df.loc[df.tick == rte, "score_first_half_ct"].iloc[0]
            rep["score_first_half_t"] = \
                df.loc[df.tick == rte, "score_first_half_t"].iloc[0]
        if ce >= 13 and ce < 25:
            rep["first_half"] = False
            rep["second_half"] = True
            rep["score_second_half_ct"] = \
                df.loc[df.tick == rte, "score_second_half_ct"].iloc[0]
            rep["score_second_half_t"] = \
                df.loc[df.tick == rte, "score_second_half_t"].iloc[0]
        rep["spotted_by"] = []
        if ce in [12, 24, 27, 30]:  rep["end_of_phase"] = True
        else:  rep["end_of_phase"] = False
        if ce == rounds[-1]:  rep["end_of_match"] = True
        else:  rep["end_of_match"] = False
        rep["attacker_steamid"] = cols_fillna[str_str]
        rep["assister_steamid"] = cols_fillna[str_str]
        rep["defender_steamid"] = cols_fillna[str_str]
        rep["is_alive"] = bnd_is_alive
    round_end_pad_df = DataFrame(list(round_end_pad_dict.values()))
    round_end_pad_df_f = round_end_pad_df.loc[
        np.repeat(round_end_pad_df.index, players_count)]\
        .reset_index(drop=True)

    round_start_pad_df_f = round_start_pad_df_f.copy()
    round_start_pad_df_f["player_name"] = \
        mgd.iloc[:players_count]["player_name"].values.tolist() * \
            len(rounds)
    round_start_pad_df_f["player_steamid"] = \
        mgd.iloc[:players_count]["player_steamid"].values.tolist() * \
            len(rounds)
    round_start_pad_df_f["is_ct"] = \
        mgd.iloc[:players_count]["is_ct"].values.tolist() * \
            len(rounds)
    round_start_pad_df_f["is_t"] = \
        mgd.iloc[:players_count]["is_t"].values.tolist() * \
            len(rounds)
    round_start_pad_df_f["player_team"] = \
        mgd.iloc[:players_count]["player_team"].values.tolist() * \
            len(rounds)
    round_start_pad_df_f = round_start_pad_df_f.copy()

    round_end_pad_df_f = round_end_pad_df_f.copy()
    round_end_pad_df_f["player_name"] = \
        mgd.iloc[:players_count]["player_name"].values.tolist() * \
            len(rounds)
    round_end_pad_df_f["player_steamid"] = \
        mgd.iloc[:players_count]["player_steamid"].values.tolist() * \
            len(rounds)
    round_end_pad_df_f["is_ct"] = \
        mgd.iloc[:players_count]["is_ct"].values.tolist() * \
            len(rounds)
    round_end_pad_df_f["is_t"] = \
        mgd.iloc[:players_count]["is_t"].values.tolist() * \
            len(rounds)
    round_end_pad_df_f["player_team"] = \
        mgd.iloc[:players_count]["player_team"].values.tolist() * \
            len(rounds)
    for rd in rounds:
        round_end_pad_df_f.loc[lambda d: d.round_total == rd,
            "won_with_ct_defused"] = mgd.loc[mgd.round_total == rd,]\
                .iloc[-players_count:]["won_with_ct_defused"]\
                    .values.tolist()
        round_end_pad_df_f.loc[lambda d: d.round_total == rd,
            "won_with_ct_delayed"] = mgd.loc[mgd.round_total == rd,]\
                .iloc[-players_count:]["won_with_ct_delayed"]\
                    .values.tolist()
        round_end_pad_df_f.loc[lambda d: d.round_total == rd,
            "won_with_ct_eliminated"] = mgd.loc[mgd.round_total == rd,]\
                .iloc[-players_count:]["won_with_ct_eliminated"]\
                    .values.tolist()
        round_end_pad_df_f.loc[lambda d: d.round_total == rd,
            "won_with_t_eliminated"] = mgd.loc[mgd.round_total == rd,]\
                .iloc[-players_count:]["won_with_t_eliminated"]\
                    .values.tolist()
        round_end_pad_df_f.loc[lambda d: d.round_total == rd,
            "won_with_t_exploded"] = mgd.loc[mgd.round_total == rd,]\
                .iloc[-players_count:]["won_with_t_exploded"]\
                    .values.tolist()
    round_end_pad_df_f = round_end_pad_df_f.copy()

    tournament = os.path.basename(os.path.dirname(demo))
    match = vs_str.join([team for team in sorted(teams)])
    map_name = header["map_name"][3:]
    map_num = np.uint8(re.search("-m[0-9]-", demo).group(0)[2:-1])

    mgd = mgd.copy()
    mgd[["start_of_round", "end_of_round"]] = False
    mgd[["start_of_phase", "end_of_phase"]] = False
    mgd[["start_of_match", "end_of_match"]] = False
    mgd[[
        "won_with_ct_defused", "won_with_ct_delayed", "won_with_ct_eliminated",
        "won_with_t_eliminated", "won_with_t_exploded",
    ]] = False
    mgd = mgd.copy()

    final = concat([round_start_pad_df_f, mgd, round_end_pad_df_f])
    final.loc[final.type.isna(), "type"] = str_str
    final = final.copy()

    final = final.drop(final.loc[(final.tick < 1),].index)
    final = final.drop(
        final.loc[(final.player_steamid == cols_fillna[str_str]),].index
    )
    final = final.drop(
        columns=["attacker_steamid", "defender_steamid", "assister_steamid"])
    final = final.sort_values(
        by=["tick", "type", "user_steamid", "player_steamid"])
    final = final.copy()

    final["timestep"] = 0
    for rd in rounds:
        rdst = final.loc[final.round_total == rd, "tick"].min()
        final.loc[final.round_total == rd, "timestep"] = \
            final.loc[final.round_total == rd, "tick"].sub(rdst)
    final["timestep"] = final["timestep"].add(1)
    final["timestep"] = final["timestep"].astype(np.int32)

    final["spotted_by"] = final["spotted_by"].astype("object")
    final["spotted_by"] = \
        final["spotted_by"].map(lambda l: [str(s) for s in l])

    final["silent"] = final["silent"].astype(np.bool_)
    final["silenced"] = final["silenced"].astype(np.bool_)

    final = final.copy()
    final[["heard_footsteps", "heard_gunfire", "seen"]] = False
    final = final.copy()
    for p in range(len(players)):
        final = final.copy()
        final.loc[
            (final.user_steamid == players[p]) &
            (final.player_steamid == players[p]) &
            (~final.silent) & (final.type == "audio_footsteps"),
        "heard_footsteps"] = True
        final.loc[
            (final.user_steamid == players[p]) &
            (final.player_steamid == players[p]) &
            (~final.silenced) & (final.type == "audio_gunfire"),
        "heard_gunfire"] = True
        final.loc[
            (final.player_steamid == players[p]) &
            (final.player_steamid == players[p]) &
            (final.spotted_by.str.len() > 0), "seen"] = True
    final = final.copy()

    for nadetyp in grenade_types:
        gc = nadetyp.replace("_", "_type_")
        gt = nadetyp.split("_")[1]
        final[f"{gc}_a"] = False
        final.loc[(final.lane_a) & (final.type == nadetyp), f"{gc}_a"] = True
        final[f"num_{gt}s_a"] = \
            final.groupby("tick")[f"{gc}_a"].transform("sum") // players_count
        final[f"{gc}_m"] = False
        final.loc[(final.lane_m) & (final.type == nadetyp), f"{gc}_m"] = True
        final[f"num_{gt}s_m"] = \
            final.groupby("tick")[f"{gc}_m"].transform("sum") // players_count
        final[f"{gc}_b"] = False
        final.loc[(final.lane_b) & (final.type == nadetyp), f"{gc}_b"] = True
        final[f"num_{gt}s_b"] = \
            final.groupby("tick")[f"{gc}_b"].transform("sum") // players_count
        final[gc] = False
        final.loc[final.type == nadetyp, gc] = True
        final[f"num_{gt}s"] = \
            final.groupby("tick")[gc].transform("sum") // players_count
        final = final.drop(columns=[f"{gc}_a", f"{gc}_m", f"{gc}_b", gc])
    final = final.copy()

    final["pitch"] = final["pitch"].add(90)
    final["pitch"] = final["pitch"].div(90)
    final["yaw"] = final["yaw"].add(180)
    final["yaw"] = final["yaw"].div(180)

    final = final.drop_duplicates(
        subset=["tick", "player_steamid"], keep="first")
    final["type"] = str_str
    final["user_steamid"] = str_str
    final = final.copy()

    final["num_players_total"] = \
        final.groupby(["tick", "type"])["is_alive"].transform("sum")
    final[["is_alive_ct", "is_alive_t"]] = False
    final.loc[(final.is_alive) & (final.is_ct), "is_alive_ct"] = True
    final.loc[(final.is_alive) & (final.is_t), "is_alive_t"] = True
    final["num_players_ct"] = \
        final.groupby(["tick", "type"])["is_alive_ct"].transform("sum")
    final["num_players_t"] = \
        final.groupby(["tick", "type"])["is_alive_t"].transform("sum")
    final = final.drop(columns=["is_alive_ct", "is_alive_t"])
    final = final.copy()

    game_idx = f"M-{tournament},{match},{map_name},{map_num:01d}"

    final["match"] = match
    for mp in active_map_names:
        final[f"map_name_{mp}"] = False
    final["map_name"] = map_name
    final[f"map_name_{map_name}"] = True
    final["map_num"] = map_num
    final["server_name"] = "_".join(header["server_name"].split(" ")[:-1])
    final["tournament"] = tournament
    for c in ["map_name", "match", "server_name", "tournament"]:
        final[c] = final[c].astype(str_str)
    final["map_num"] = final["map_num"].astype(np.uint8)
    final["game_idx"] = game_idx
    final = final.copy()

    final = final.reset_index(drop=True)
    for c in final.columns:
        for k in ["situation", "round", "game", "match"]:
            if c in cols["node"][k]:
                t = cols["node"][k][c]
                final[c] = final[c].fillna(cols_fillna[t]).astype(t)
            if c in cols["downsample"][k]:
                t = cols["downsample"][k][c]
                final[c] = final[c].fillna(cols_fillna[t]).astype(t)
            if c in cols["graph"][k]:
                if k == "game":
                    t = cols["graph"][k][c]
                    final[c] = final[c].fillna(cols_fillna[t]).astype(t)
    final["game_idx"] = final["game_idx"].astype("string")
    final["tick"] = final["tick"].astype("int64")
    final["player_steamid"] = final["player_steamid"].astype("string")
    final = final.copy()

    final = final.drop(columns=[
        c for c in list(cols["graph"]["situation"].keys()) if c != "tick"
    ])
    final = final.drop(columns=(
        list(cols["graph"]["round"].keys()) +
        list(cols["graph"]["match"].keys())
    ))
    final = final.drop(columns=["map_name", "player_name", "spotted_by"])
    final = final.copy()

    final = final.sort_values(by=list(index_cols.keys()))
    final = final.set_index(list(index_cols.keys()), drop=True)
    final = final.copy()

    if parse_first_half and parse_second_half:
        final = final.drop(
            final.loc[(~final.first_half) & (~final.second_half),].index
        )
    if parse_first_half and not parse_second_half:
        final = final.drop(final.loc[(~final.first_half),].index)
    if not parse_first_half and parse_second_half:
        final = final.drop(final.loc[(~final.second_half),].index)
    if not parse_first_half and not parse_second_half:
        pass
    final = final.copy()
    final.to_pickle(os.path.join(frame_dir, f"{game_idx}.{pickle_str}"))

    return (final.copy(), game_idx)


def load_dataframe(saved: str) -> tuple[DataFrame, str]:

    df = read_pickle(os.path.join(frame_dir, f"{saved}.{pickle_str}"))

    return (df.copy(), saved)


def tensorize_dataframe(final: DataFrame, save: str) -> tuple[list, list]:

    final = final.drop(columns=["first_half", "second_half"])

    final = final.drop(
        columns=["start_of_match", "start_of_phase", "start_of_round"])
    final = final.drop(
        columns=["end_of_round", "end_of_match", "end_of_phase"])

    final = final.drop(columns=["player_team"])

    final["reward_ct"] = \
        final.won_with_ct_defused * ct_defused_factor + \
        final.won_with_ct_delayed * ct_delayed_factor + \
        final.won_with_ct_eliminated * ct_eliminated_factor + \
        0.0
    final["reward_t"] = \
        final.won_with_t_exploded * t_exploded_factor + \
        final.won_with_t_eliminated * t_eliminated_factor + \
        0.0

    final = final.drop(columns=[
        "won_with_ct_defused", "won_with_ct_delayed", "won_with_ct_eliminated",
    ])
    final = final.drop(columns=[
        "won_with_t_exploded", "won_with_t_eliminated",
    ])

    col_list_teams = [
        c for c in list(final.columns) if c not in ["x_d", "y_d", "z_d"]
    ]
    col_list = [
        c for c in list(final.columns)
        if c not in ["x_d", "y_d", "z_d"] and c not in ["is_ct", "is_t"]
    ]
    col_list = [c for c in col_list if c != "player_steamid"]

    rounds = list(factorize(final.round_total)[1])
    ct_actions, t_actions = [], []
    ct_obsvs, t_obsvs = [], []
    ct_infos, t_infos = [], []
    ct_dones, t_dones = [], []
    ct_rewards, t_rewards = [], []
    for rd in rounds:
        final_rd = final.loc[final.round_total == rd,].copy()
        game_idx = final_rd.index.get_level_values(0)
        final_rd["player_steamid"] = final_rd.index.get_level_values(2)
        players = list(factorize(final_rd.player_steamid)[1])
        final_rds_ct, final_rds_t = [], []
        actions_ct, actions_t = [], []
        obsvs_ct, obsvs_t = [], []
        infos_ct, infos_t = [], []
        dones_ct, dones_t = [], []
        rewards_ct, rewards_t = [], []
        for player in players:
            f = final_rd.loc[final_rd.player_steamid == player,].copy()
            f = f.drop(columns=["player_steamid"])
            f = f[col_list_teams].copy()
            if all(f.is_ct.tolist()):  final_rds_ct.append(f)
            if all(f.is_t.tolist()):  final_rds_t.append(f)
        for fct in final_rds_ct:
            fct = fct[col_list].copy()
            if rd >= 1 and rd < 13:
                fct["score_difference"] = \
                    fct["score_first_half_ct"] - fct["score_first_half_t"]
            if rd >= 13 and rd < 25:
                fct["score_difference"] = \
                    fct["score_second_half_ct"] - fct["score_second_half_t"]
            fct = fct.astype(np.float32)
            actions_ct.append(fct[action_colnames])
            obsvs_ct.append(fct[[
                c for c in col_list if c not in action_colnames
                and "reward" not in c and "score" not in c and "round" not in c
                and c != "is_alive"
            ] + ["score_difference"]])
            infos_ct.append(fct[["round_total", "round_phase"]])
            rewards_ct.append(fct[["reward_ct"]])
            dones_ct.append(fct[["is_alive"]])
        for ft in final_rds_t:
            ft = ft[col_list].copy()
            if rd >= 1 and rd < 13:
                ft["score_difference"] = \
                    ft["score_first_half_t"] - ft["score_first_half_ct"]
            if rd >= 13 and rd < 25:
                ft["score_difference"] = \
                    ft["score_second_half_t"] - ft["score_second_half_ct"]
            ft = ft.astype(np.float32)
            actions_t.append(ft[action_colnames])
            obsvs_t.append(ft[[
                c for c in col_list if c not in action_colnames
                and "reward" not in c and "score" not in c and "round" not in c
                and c != "is_alive"
            ] + ["score_difference"]])
            infos_t.append(ft[["round_total", "round_phase"]])
            rewards_t.append(ft[["reward_t"]])
            dones_t.append(ft[["is_alive"]])

        ct_actions.append(torch.stack(
            [torch.tensor(f.values, dtype=torch.float32) for f in actions_ct],
            dim=1
        ))
        ct_obsvs.append(torch.stack(
            [torch.tensor(f.values, dtype=torch.float32) for f in obsvs_ct],
            dim=1
        ))
        ct_infos.append(torch.stack(
            [torch.tensor(f.values, dtype=torch.float32) for f in infos_ct],
            dim=1
        ))
        ct_rewards.append(torch.stack(
            [torch.tensor(f.values, dtype=torch.float32) for f in rewards_ct],
            dim=1
        ))
        ct_dones.append(torch.stack(
            [torch.tensor(f.values, dtype=torch.float32) for f in dones_ct],
            dim=1
        ))
        t_actions.append(torch.stack(
            [torch.tensor(f.values, dtype=torch.float32) for f in actions_t],
            dim=1
        ))
        t_obsvs.append(torch.stack(
            [torch.tensor(f.values, dtype=torch.float32) for f in obsvs_t],
            dim=1
        ))
        t_infos.append(torch.stack(
            [torch.tensor(f.values, dtype=torch.float32) for f in infos_t],
            dim=1
        ))
        t_rewards.append(torch.stack(
            [torch.tensor(f.values, dtype=torch.float32) for f in rewards_t],
            dim=1
        ))
        t_dones.append(torch.stack(
            [torch.tensor(f.values, dtype=torch.float32) for f in dones_t],
            dim=1
        ))

    ct_dicts, t_dicts = [], []
    for rd in rounds:
        ct_ep_dict = {
            "actions": ct_actions[rd-1].numpy(),
            "observations": ct_obsvs[rd-1].numpy(),
            "informations": ct_infos[rd-1].numpy(),
            "rewards": ct_rewards[rd-1].numpy(),
            "dones": ct_dones[rd-1].numpy(),
        }
        ct_dicts.append(ct_ep_dict)
        t_ep_dict = {
            "actions": t_actions[rd-1].numpy(),
            "observations": t_obsvs[rd-1].numpy(),
            "informations": t_infos[rd-1].numpy(),
            "rewards": t_rewards[rd-1].numpy(),
            "dones": t_dones[rd-1].numpy(),
        }
        t_dicts.append(t_ep_dict)

    np.savez(
        os.path.join(npz_dir, f"{save}_ct.{npz_str}"),
        episodes=ct_dicts
    )
    np.savez(
        os.path.join(npz_dir, f"{save}_t.{npz_str}"),
        episodes=t_dicts
    )

    return (ct_dicts, t_dicts)


def run_datum(
    mch: str, saved_frame: str="", saved_npz: str="", half: int=0
) -> tuple[list, list]:

    mp = mch
    if mp != "":
        if not saved_frame:
            df, saved_frame = \
                parse_demo(
                    mp,
                    parse_first_half=(half == 1), parse_second_half=(half == 2)
                )
        else:
            df, saved_frame = load_dataframe(saved_frame)
        if df.shape[0] > 0 and df.shape[0] % players_count == 0:
            if not saved_npz:
                ct_dicts, t_dicts = tensorize_dataframe(df, saved_frame)
                saved = saved_frame
            else:
                ct_dicts = np.load(
                    os.path.join(npz_dir, f"{saved_npz}_ct.npz"),
                    allow_pickle=True
                )
                ct_dicts = ct_dicts["episodes"].tolist()
                t_dicts = np.load(
                    os.path.join(npz_dir, f"{saved_npz}_t.npz"),
                    allow_pickle=True
                )
                t_dicts = t_dicts["episodes"].tolist()
                saved = saved_npz

    return (ct_dicts, t_dicts)


def run_data(files: list[str], saved: bool, tag: str, half: int) -> None:

    ct_ds, t_ds = [], []
    for mch in files[:]:
        if not saved:
            ct_d, t_d = run_datum(mch, saved_frame="", saved_npz="", half=half)
        else:
            ct_d, t_d = run_datum(None, saved_frame=mch, saved_npz=mch, half=0)
        ct_ds += ct_d
        t_ds += t_d

    np.savez(
        os.path.join(cloud_dir, f"ct_{tag}_rounds.{npz_str}"),
        episodes=ct_ds,
    )
    np.savez(
        os.path.join(cloud_dir, f"t_{tag}_rounds.{npz_str}"),
        episodes=t_ds,
    )

    return None


class CS2OfflineEnv(MultiAgentEnv):

    def __init__(self, config) -> None:

        super().__init__()

        data_path = config["data_path"]
        data = np.load(data_path, allow_pickle=True)

        self.episodes = data["episodes"]
        self.idx = -1
        self.t = -1

        self._agent_ids = [f"agent_{i+1}" for i in range(team_count)]
        self.possible_agents = [f"agent_{i+1}" for i in range(team_count)]
        self.agents = [f"agent_{i+1}" for i in range(team_count)]

        self.observation_space = spaces.Dict({
            f"agent_{i+1}": spaces.Box(
                low=-np.inf, high=+np.inf, shape=(217,), dtype=np.float32,
            ) for i in range(team_count)
        })

        self.action_space = spaces.Dict({
            f"agent_{i+1}": spaces.Box(
                low=0.0, high=1.0, shape=(9,), dtype=np.float32
            ) for i in range(team_count)
        })


    def reset(self, *, seed=None, options=None) -> tuple[dict, dict]:

        self.idx = ((self.idx + 1) % len(self.episodes))

        self.t = 0
        self.agents = [f"agent_{i+1}" for i in range(team_count)]

        ep = self.episodes[self.idx]

        obsvs = {
            f"agent_{i+1}":
            ep["observations"][self.t][i] for i in range(team_count)
        }
        info_vec = ep["informations"][self.t]
        infos = {
            f"agent_{i+1}": {
                "round_total": np.uint8(info_vec[i][0]),
                "round_phase": np.uint8(info_vec[i][1]),
            } for i in range(team_count)
        }

        return (obsvs, infos)


    def step(self, action) -> tuple[dict, dict, dict, dict, dict]:

        ep = self.episodes[self.idx]
        truncs = {k: False for k in self.possible_agents}

        dones = {
            f"agent_{i+1}": not bool(ep["dones"][self.t][i][0])
            for i in range(team_count)
        }
        if self.t > 0:
            self.agents = [k for k, v in dones.items() if not v]
        else:
            self.agents = [f"agent_{i+1}" for i in range(team_count)]
        dones["__all__"] = False
        rewards = {
            f"agent_{i+1}": np.float32(ep["rewards"][self.t][i][0])
            for i in range(team_count) if f"agent_{i+1}" in self.agents
        }

        done = self.t == (len(ep["dones"]) - 1)
        dones["__all__"] = done
        if not done:
            self.t += 1
            obsvs = {
                f"agent_{i+1}":
                ep["observations"][self.t][i] for i in range(team_count)
                if f"agent_{i+1}" in self.agents
            }
            info_vec = ep["informations"][self.t]
            infos = {
                f"agent_{i+1}": {
                    "round_total": np.uint8(info_vec[i][0]),
                    "round_phase": np.uint8(info_vec[i][1]),
                } for i in range(team_count) if f"agent_{i+1}" in self.agents
            }
        else:
            obsvs, infos = {}, {}

        return obsvs, rewards, dones, truncs, infos


class CS2OfflineEnvCT(MultiAgentEnv):

    def __init__(self, config) -> None:

        super().__init__()

        data_path = config["data_path"]
        data = np.load(data_path, allow_pickle=True)

        self.episodes = data["episodes"]
        self._has_probed = False
        self.idx = -1
        self.t = -1

        self._agent_ids = [f"agent_{i+1}" for i in range(team_count)]
        self.possible_agents = [f"agent_{i+1}" for i in range(team_count)]
        self.agents = [f"agent_{i+1}" for i in range(team_count)]

        self.observation_space = spaces.Dict({
            f"agent_{i+1}": spaces.Box(
                low=-np.inf, high=+np.inf, shape=(217,), dtype=np.float32,
            ) for i in range(team_count)
        })

        self.action_space = spaces.Dict({
            f"agent_{i+1}": spaces.Box(
                low=0.0, high=1.0, shape=(9,), dtype=np.float32
            ) for i in range(team_count)
        })


    def reset(self, *, seed=None, options=None) -> tuple[dict, dict]:

        if not self._has_probed:
            self._has_probed = True
            self.idx = 0
        else:
            self.idx = ((self.idx + 1) % len(self.episodes))

        self.t = 0
        self.agents = [f"agent_{i+1}" for i in range(team_count)]

        ep = self.episodes[self.idx]

        obsvs = {
            f"agent_{i+1}":
            ep["observations"][self.t][i] for i in range(team_count)
        }
        info_vec = ep["informations"][self.t]
        infos = {
            f"agent_{i+1}": {
                "round_total": np.uint8(info_vec[i][0]),
                "round_phase": np.uint8(info_vec[i][1]),
            } for i in range(team_count)
        }

        return (obsvs, infos)


    def step(self, action) -> tuple[dict, dict, dict, dict, dict]:

        ep = self.episodes[self.idx]
        truncs = {k: False for k in self.possible_agents}

        dones = {
            f"agent_{i+1}": not bool(ep["dones"][self.t][i][0])
            for i in range(team_count)
        }
        if self.t > 0:
            self.agents = [k for k, v in dones.items() if not v]
        else:
            self.agents = [f"agent_{i+1}" for i in range(team_count)]
        dones["__all__"] = False
        rewards = {
            f"agent_{i+1}": np.float32(ep["rewards"][self.t][i][0])
            for i in range(team_count) if f"agent_{i+1}" in self.agents
        }

        done = self.t == (len(ep["dones"]) - 1)
        dones["__all__"] = done
        if not done:
            self.t += 1
            obsvs = {
                f"agent_{i+1}":
                ep["observations"][self.t][i] for i in range(team_count)
                if f"agent_{i+1}" in self.agents
            }
            info_vec = ep["informations"][self.t]
            infos = {
                f"agent_{i+1}": {
                    "round_total": np.uint8(info_vec[i][0]),
                    "round_phase": np.uint8(info_vec[i][1]),
                } for i in range(team_count) if f"agent_{i+1}" in self.agents
            }
        else:
            obsvs, infos = {}, {}

        return obsvs, rewards, dones, truncs, infos


class CS2OfflineEnvT(MultiAgentEnv):

    def __init__(self, config) -> None:

        super().__init__()

        data_path = config["data_path"]
        data = np.load(data_path, allow_pickle=True)

        self.episodes = data["episodes"]
        self._has_probed = False
        self.idx = -1
        self.t = -1

        self._agent_ids = [f"agent_{i+1}" for i in range(team_count)]
        self.possible_agents = [f"agent_{i+1}" for i in range(team_count)]
        self.agents = [f"agent_{i+1}" for i in range(team_count)]

        self.observation_space = spaces.Dict({
            f"agent_{i+1}": spaces.Box(
                low=-np.inf, high=+np.inf, shape=(217,), dtype=np.float32,
            ) for i in range(team_count)
        })

        self.action_space = spaces.Dict({
            f"agent_{i+1}": spaces.Box(
                low=0.0, high=1.0, shape=(9,), dtype=np.float32
            ) for i in range(team_count)
        })


    def reset(self, *, seed=None, options=None) -> tuple[dict, dict]:

        if not self._has_probed:
            self._has_probed = True
            self.idx = 0
        else:
            self.idx = ((self.idx + 1) % len(self.episodes))

        self.t = 0
        self.agents = [f"agent_{i+1}" for i in range(team_count)]

        ep = self.episodes[self.idx]

        obsvs = {
            f"agent_{i+1}":
            ep["observations"][self.t][i] for i in range(team_count)
        }
        info_vec = ep["informations"][self.t]
        infos = {
            f"agent_{i+1}": {
                "round_total": np.uint8(info_vec[i][0]),
                "round_phase": np.uint8(info_vec[i][1]),
            } for i in range(team_count)
        }

        return (obsvs, infos)


    def step(self, action) -> tuple[dict, dict, dict, dict, dict]:

        ep = self.episodes[self.idx]
        truncs = {k: False for k in self.possible_agents}

        dones = {
            f"agent_{i+1}": not bool(ep["dones"][self.t][i][0])
            for i in range(team_count)
        }
        if self.t > 0:
            self.agents = [k for k, v in dones.items() if not v]
        else:
            self.agents = [f"agent_{i+1}" for i in range(team_count)]
        dones["__all__"] = False
        rewards = {
            f"agent_{i+1}": np.float32(ep["rewards"][self.t][i][0])
            for i in range(team_count) if f"agent_{i+1}" in self.agents
        }

        done = self.t == (len(ep["dones"]) - 1)
        dones["__all__"] = done
        if not done:
            self.t += 1
            obsvs = {
                f"agent_{i+1}":
                ep["observations"][self.t][i] for i in range(team_count)
                if f"agent_{i+1}" in self.agents
            }
            info_vec = ep["informations"][self.t]
            infos = {
                f"agent_{i+1}": {
                    "round_total": np.uint8(info_vec[i][0]),
                    "round_phase": np.uint8(info_vec[i][1]),
                } for i in range(team_count) if f"agent_{i+1}" in self.agents
            }
        else:
            obsvs, infos = {}, {}

        return obsvs, rewards, dones, truncs, infos


def make_cs2_ct_env(cfg):

    return CS2OfflineEnvCT(cfg)

register_env("cs2_ct_env", make_cs2_ct_env)


def make_cs2_t_env(cfg):

    return CS2OfflineEnvT(cfg)

register_env("cs2_t_env", make_cs2_t_env)


def make_cs2_env(cfg):

    return CS2OfflineEnv(cfg)

register_env("cs2_env", make_cs2_env)


def run_algo(mch: str, team: str) -> ray.rllib.algorithms.ppo.ppo.PPO:

    if team == "ct":
        tm_env = "cs2_ct_env"
    if team == "t":
        tm_env = "cs2_t_env"
    if team != "ct" and team != "t":
        tm_env = "cs2_env"

    config = (
        RLConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .framework("torch")
        .resources(num_gpus=1)
        .environment(
            env=tm_env, # distinct ray environment names just in case
            env_config={"data_path": mch},
        )
        .env_runners(
            num_env_runners=5,
            batch_mode="complete_episodes",
            # explore=True,
        )
        .offline_data(input_=[mch])
        .evaluation(
            evaluation_interval=1,
            evaluation_duration=1,
            evaluation_duration_unit="episodes"
        )
        .training(model={
            "uses_new_env_runners": True,
            "use_gru": True,
            "gru_cell_size": 256,
            "max_seq_len": 50,
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
            "normalize_observations": True,
            "action_dist": "tanh_normal",
            "squash_to_range": True,
        })
        .multi_agent(
            policies={
                "p_agent_1", "p_agent_2", "p_agent_3", "p_agent_4", "p_agent_5"
            },
            policy_mapping_fn=lambda agent_id, episode, **kw: f"p_{agent_id}",
            policies_to_train=[
                "p_agent_1", "p_agent_2", "p_agent_3", "p_agent_4", "p_agent_5"
            ],
        )
    )

    logs = []
    algo = config.build()
    logs = algo.train()
    if team == "ct":  ckpt = algo.save(os.path.join(logs_ct_dir))
    if team == "t":  ckpt = algo.save(os.path.join(logs_t_dir))
    if team != "ct" and team != "t":  algo.save(os.path.join(logs_dir))

    return algo


def generate_actions(
    mch: str, team: str, round_timesteps: list[tuple[int, int]]
) -> list[dict[str, np.ndarray]]:

    env = CS2OfflineEnv({"data_path": mch})
    if team == "ct":
        algo = Algorithm.from_checkpoint(os.path.join(logs_ct_dir))
    if team == "t":
        algo = Algorithm.from_checkpoint(os.path.join(logs_t_dir))
    if team != "ct" and team != "t":
        algo = Algorithm.from_checkpoint(os.path.join(logs_dir))
    modules = [
        algo.get_module(pid)
            for pid in [
                "p_agent_1", "p_agent_2", "p_agent_3", "p_agent_4", "p_agent_5"
        ]
    ]
    round_timesteps = [
        rdt for rdt in round_timesteps
        if rdt[0] > 0 and rdt[0] < 13 and rdt[1] > 1
    ]

    generated_actions = []
    for rdt in round_timesteps:
        env.idx = rdt[0] - 1
        obsvs, infos = env.reset()
        if rdt[-1] > len(env.episodes[rdt[0] - 1]) - 2:
            env.t = len(env.episodes[rdt[0] - 1]) - 2
        else:
            env.t = rdt[1] - 1
        obsvs, _, _, _, infos = env.step(None)
        generated_action = {}
        for aid in ["agent_1", "agent_2", "agent_3", "agent_4", "agent_5"]:
            obs = obsvs[aid]
            batch = {"obs": torch.from_numpy(obs).unsqueeze(0)}
            dist_cls = modules[0].get_inference_action_dist_cls()
            out = modules[0].forward_inference(batch)
            dist = dist_cls.from_logits(out["action_dist_inputs"])
            raw_action = dist.sample()
            new_action = ((raw_action + 1.0) / 2.0)
            clamped_action = new_action.clamp(0.0, 1.0)
            agent_action = clamped_action.squeeze(0).cpu().numpy()
            agent_action[:7] = agent_action[:7].round()
            agent_action[7] = (agent_action[7] * 90) - 90
            agent_action[8] = (agent_action[8] * 180) - 180
            generated_action[aid] = agent_action
        generated_actions.append(generated_action)
        obsvs, _, _, _, infos = env.step(generated_action)

    return generated_actions


"""MAIN"""


def main():

    print("\n\nMain.\n\n")
    t0 = perf_counter()

    # only PGL Copenhagen 2024 data has been tested in this experiment
    tournament_str = "tournament_name_0"

    demofiles = retrieve_demo_paths(
        tournaments=[tournament_str, ],
        teams=[],
        matches=[],
    )[:]
    demofiles = sorted(demofiles, key=lambda l: l[0])

    run_data(demofiles, False, tournament_str, 1)

    npzfiles = [ # cannot derive these strings
        "placeholder_demo_0",
    ]
    run_data(npzfiles, True, tournament_str, 1)

    algo = {}
    for tm in ["ct", "t"][:]:
        algo[tm] = run_algo(
            os.path.join(cloud_dir, f"{tm}_{tournament_str}_rounds.{npz_str}"),
            tm
        )

    # {team: {agent: action tensor for 5 agents} for 2 teams}
    actions = {}
    for tm in ["ct", "t"][:]:
        actions[tm] = generate_actions(
            os.path.join(cloud_dir, f"{tm}_{tournament_str}_rounds.{npz_str}"),
            tm, # input observation data: [(round, timestep)] ->
            [(1, 2), (3, tick_count * 30), (12, 1e8)]
        )
    # output action data: ->
    # action => [back, forward, left, right, fire, reload, use, pitch, yaw]
    # action => [S, W, A, D, LEFTCLICK, R, E or F or G, vertical, horizontal]
    # action => [7 booleans followed by 2 floats]

    t1 = perf_counter()
    print(f"\n\nRan for {(t1 - t0):0.3f} seconds.\n\n")
    print("\n\nDone.\n\n")

    return None


if __name__ == "__main__":  main()


# TODO  pass


