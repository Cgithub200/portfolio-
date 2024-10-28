using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace VertoTest.Models
{
    public class CalendarEvent
    {
        [Key]
        [DatabaseGenerated(DatabaseGeneratedOption.Identity)]
        public int EventID { get; set; }

        public required DateTime EventStart { get; set; }
        public required DateTime EventEnd { get; set; }

        public string? Title { get; set; }

        public string? Description { get; set; }

        public required DateTime LastUpdated { get; set; }


    }
}
