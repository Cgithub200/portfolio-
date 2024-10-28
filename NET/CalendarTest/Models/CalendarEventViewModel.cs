using Microsoft.Extensions.Logging;
using System.ComponentModel.DataAnnotations;

namespace VertoTest.Models
{
    public class CalendarEventViewModel
    {

        public required int EventID { get; set; }

        public required DateTime EventStart { get; set; }
        public required DateTime EventEnd { get; set; }

        public string? Title { get; set; }

        public string? Description { get; set; }


    }


}
