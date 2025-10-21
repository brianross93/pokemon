import unittest

from src.pkmn_battle.env.blue_ram_adapter import (
    BlueBattleRAMMap,
    BlueRAMAdapter,
    FieldSpec,
    MoveSlotSpec,
    PartySpec,
    PokemonSlotSpec,
)


class TestBlueRAMAdapter(unittest.TestCase):
    def test_reads_party_and_field(self) -> None:
        memory = {
            0x1000: 3,  # turn_counter
            0x1001: 25,  # my active species
            0x1002: 42,  # level
            0x1003: 0x00,
            0x1004: 120,  # HP lo
            0x1005: 0x00,
            0x1006: 140,
            0x1007: 2,  # status
            0x1008: 85,
            0x1009: 10,
            0x100A: 15,
            0x100B: 0,
            0x100C: 86,
            0x100D: 5,
            0x100E: 10,
            0x100F: 0,
            0x1010: 87,
            0x1011: 3,
            0x1012: 10,
            0x1013: 0,
            0x1014: 88,
            0x1015: 2,
            0x1016: 5,
            0x1017: 0,
            0x1100: 7,  # party slot 1 species
            0x1101: 30,
            0x1102: 0,
            0x1103: 80,
            0x1104: 0,
            0x1105: 90,
            0x1106: 91,
            0x1107: 12,
            0x1110: 8,  # party slot 2 species
            0x1111: 28,
            0x1112: 0,
            0x1113: 60,
            0x1114: 0,
            0x1115: 72,
            0x1116: 92,
            0x1117: 14,
            0x1200: 1,  # weather
            0x1201: 0b11,  # player screens
            0x1202: 0b01,  # enemy screens
        }

        ram_map = BlueBattleRAMMap(
            turn_counter=0x1000,
            my_active=PokemonSlotSpec(
                species=0x1001,
                level=0x1002,
                current_hp=(0x1003, 0x1004),
                max_hp=(0x1005, 0x1006),
                status=0x1007,
                moves=(
                    MoveSlotSpec(move_id=0x1008, pp=0x1009, max_pp=0x100A, disabled_flag=0x100B),
                    MoveSlotSpec(move_id=0x100C, pp=0x100D, max_pp=0x100E, disabled_flag=0x100F),
                    MoveSlotSpec(move_id=0x1010, pp=0x1011, max_pp=0x1012, disabled_flag=0x1013),
                    MoveSlotSpec(move_id=0x1014, pp=0x1015, max_pp=0x1016, disabled_flag=0x1017),
                ),
            ),
            opp_active=PokemonSlotSpec(),
            my_party=PartySpec(
                count=2,
                slots=(
                    PokemonSlotSpec(
                        species=0x1100,
                        level=0x1101,
                        current_hp=(0x1102, 0x1103),
                        max_hp=(0x1104, 0x1105),
                        moves=(MoveSlotSpec(move_id=0x1106, pp=0x1107),),
                    ),
                    PokemonSlotSpec(
                        species=0x1110,
                        level=0x1111,
                        current_hp=(0x1112, 0x1113),
                        max_hp=(0x1114, 0x1115),
                        moves=(MoveSlotSpec(move_id=0x1116, pp=0x1117),),
                    ),
                ),
            ),
            opp_party=None,
            field=FieldSpec(
                weather=0x1200,
                player_screens=0x1201,
                enemy_screens=0x1202,
            ),
        )

        adapter = BlueRAMAdapter(read_u8=lambda addr: memory.get(addr, 0), ram_map=ram_map)
        obs = adapter.observe()

        self.assertEqual(obs["turn"], 3)
        self.assertEqual(obs["my_active"]["species"], 25)
        self.assertEqual(obs["my_active"]["hp"], (0 << 8) | 120)
        self.assertEqual(len(obs["my_party"]), 2)
        self.assertEqual(obs["my_party"][0]["species"], 7)
        self.assertEqual(obs["my_party"][1]["moves"][0]["pp"], 14)
        self.assertIn("weather_id", obs["field"])
        self.assertEqual(obs["field"]["weather_id"], 1)
        self.assertEqual(obs["field"]["player_screens"], 0b11)
        self.assertEqual(obs["field"]["enemy_screens"], 0b01)


if __name__ == "__main__":
    unittest.main()

